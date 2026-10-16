// VoxLat Task 12 - build the jacket with PicoGK, write STL, voxel masks and a JSON report.
//
// Entry points (called from Python through pythonnet, see voxlat/export/picogk.py):
//   JacketBuilder.RunWithViewer(configPath, logDir, keepViewerOpen)  - Library.Go pattern (viewer window)
//   JacketBuilder.RunHeadless(configPath)                             - new Library(voxel) without viewer
//   JacketBuilder.Run(configPath)                                     - inside an already running library
// Pipeline: implicit -> Voxels (wide sector in sector mode) -> raw mask -> sliver opening (Offset -r, +r)
//           -> clean mask -> trim to the nominal sector -> Mesh -> STL -> report.

using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Numerics;
using System.Text;
using System.Text.Json;
using System.Threading;
using PicoGK;

namespace VoxLat.Export
{
    public static class JacketBuilder
    {
        public static void Log(string msg)
        {
            string line = $"[VoxLat.Export] {msg}";
            Console.WriteLine(line);
            try
            {
                Library.Log(line);
            }
            catch (Exception)
            {
                // headless mode has no PicoGK log
            }
        }

        /// <summary>Run with the PicoGK viewer (Library.Go). Returns the report path.</summary>
        public static string RunWithViewer(string configPath, string logDir, bool keepViewerOpen)
        {
            ExportConfig cfg = ExportConfig.Load(configPath);
            string result = "";
            Exception? error = null;
            Directory.CreateDirectory(logDir);
            Library.Go((float)cfg.VoxelSize,
                       () =>
                       {
                           try
                           {
                               result = Run(configPath);
                           }
                           catch (Exception e)
                           {
                               error = e;
                               Log("FAILED: " + e);
                           }
                       },
                       logDir, "voxlat_export.log", Path.GetDirectoryName(configPath) ?? ".", "",
                       !keepViewerOpen);
            if (error != null)
                throw new InvalidOperationException("VoxLat export failed: " + error.Message, error);
            return result;
        }

        /// <summary>Run without viewer (PicoGK headless library). Returns the report path.</summary>
        public static string RunHeadless(string configPath)
        {
            ExportConfig cfg = ExportConfig.Load(configPath);
            using (Library lib = new Library((float)cfg.VoxelSize))
            {
                return Run(configPath);
            }
        }

        /// <summary>Build inside a running PicoGK library (voxel size = Library.fVoxelSizeMM).</summary>
        public static string Run(string configPath)
        {
            Stopwatch total = Stopwatch.StartNew();
            ExportConfig cfg = ExportConfig.Load(configPath);
            double h = Library.fVoxelSizeMM;
            if (Math.Abs(h - cfg.VoxelSize) > 1e-6 * Math.Max(1.0, cfg.VoxelSize))
                Log($"WARNING: library voxel size {h} mm differs from the config ({cfg.VoxelSize} mm); using {h} mm");
            Directory.CreateDirectory(cfg.OutputDir);
            string stem = Path.Combine(cfg.OutputDir, cfg.Name);

            Stopwatch sw = Stopwatch.StartNew();
            JacketField field = new JacketField(cfg);
            double tTables = sw.Elapsed.TotalSeconds;
            Log($"design '{field.Fields.Label}' ({cfg.Layout}), mode {cfg.Mode}, voxel {h} mm; field {field.DescribeJson()}");

            double renderHalf = field.Sector ? field.SectorHalfWide : -1.0;
            GradedTpmsJacket imp = new GradedTpmsJacket(field, renderHalf);
            BBox3 box = imp.oBounds;
            Vector3 ext = box.vecMax - box.vecMin;
            double nVox = (double)(ext.X / h) * (ext.Y / h) * (ext.Z / h);
            Log($"rendering implicit over {ext.X:F1} x {ext.Y:F1} x {ext.Z:F1} mm ({nVox / 1e6:F0} M voxels) ...");
            sw.Restart();
            Voxels vox = new Voxels(imp, box);
            double tRender = sw.Elapsed.TotalSeconds;
            vox.CalculateProperties(out float volRaw, out BBox3 bbRaw);
            Log($"rendered in {tRender:F1} s, raw volume {volRaw:F1} mm^3");

            string frame = FrameJson(field, renderHalf);
            double tMaskRaw = 0.0;
            if (cfg.WriteRawMask)
            {
                sw.Restart();
                WriteMask(vox, stem + "_mask_raw", h, frame, "raw", field);
                tMaskRaw = sw.Elapsed.TotalSeconds;
            }

            // sliver opening (Offset -r, +r); with a band only near the open lattice faces, where clipping leaves
            // sub-min-wall slivers: vox = (vox outside band) + (opened vox inside band)
            double r = cfg.SliverRadiusMm;
            double band = cfg.SliverBandMm;
            sw.Restart();
            if (r > 0.0)
            {
                if (band > 0.0)
                {
                    Voxels opened = new Voxels(vox);
                    opened.Offset(-(float)r);
                    opened.Offset((float)r);
                    opened.IntersectImplicit(new InterfaceBandImplicit(field, band, true));
                    vox.IntersectImplicit(new InterfaceBandImplicit(field, band, false));
                    vox.BoolAdd(opened);
                }
                else
                {
                    vox.Offset(-(float)r);
                    vox.Offset((float)r);
                }
            }
            double tOpen = sw.Elapsed.TotalSeconds;
            vox.CalculateProperties(out float volClean, out BBox3 bbClean);
            Log($"sliver opening r = {r} mm (band {band} mm): volume {volRaw:F1} -> {volClean:F1} mm^3 ({tOpen:F1} s)");
            double tMask = 0.0;
            if (cfg.WriteMask)
            {
                sw.Restart();
                WriteMask(vox, stem + "_mask", h, frame, "clean", field);
                tMask = sw.Elapsed.TotalSeconds;
            }

            Voxels voxOut = vox;
            if (field.Sector)
            {
                voxOut = new Voxels(vox);
                voxOut.IntersectImplicit(new WedgeImplicit(field.SectorHalf));
            }
            voxOut.CalculateProperties(out float volOut, out BBox3 bbOut);

            int tris = 0;
            string stl = "";
            double tStl = 0.0;
            if (cfg.SaveStl)
            {
                sw.Restart();
                Mesh msh = new Mesh(voxOut);
                tris = msh.nTriangleCount();
                stl = stem + ".stl";
                msh.SaveToStlFile(stl);
                tStl = sw.Elapsed.TotalSeconds;
                Log($"STL {stl}: {tris} triangles ({tStl:F1} s)");
            }
            if (cfg.ShowInViewer)
            {
                try
                {
                    Library.oViewer().Add(voxOut);
                }
                catch (Exception)
                {
                    // headless: no viewer
                }
            }

            string report = stem + "_report.json";
            using (MemoryStream ms = new MemoryStream())
            {
                using (Utf8JsonWriter w = new Utf8JsonWriter(ms, new JsonWriterOptions { Indented = true }))
                {
                    w.WriteStartObject();
                    w.WriteString("schema", "voxlat-export-report/1");
                    w.WriteString("config", cfg.ConfigPath);
                    w.WriteString("design", cfg.DesignPath);
                    w.WriteString("design_label", field.Fields.Label);
                    w.WriteString("layout", cfg.Layout);
                    w.WriteString("mode", cfg.Mode);
                    w.WriteString("picogk_version", SafeVersion());
                    w.WriteNumber("voxel_mm", h);
                    w.WriteNumber("sector_center_deg", cfg.SectorCenterDeg);
                    w.WriteNumber("sector_angle_deg", cfg.SectorAngleDeg);
                    w.WriteNumber("render_half_angle_deg", renderHalf < 0 ? -1.0 : renderHalf * 180.0 / Math.PI);
                    w.WriteString("stl_frame", field.Sector
                        ? "sector-local: world rotated by -sector_center_deg about z (sector centred on +x)"
                        : "world: z = motor axis, z = 0 at the start of the heated length, x = inlet manifold");
                    w.WritePropertyName("render_box_mm");
                    WriteBox(w, box);
                    w.WriteNumber("volume_raw_mm3", volRaw);
                    w.WriteNumber("volume_clean_mm3", volClean);
                    w.WriteNumber("volume_output_mm3", volOut);
                    w.WriteNumber("sliver_radius_mm", r);
                    w.WriteNumber("sliver_band_mm", band);
                    w.WritePropertyName("output_bbox_mm");
                    WriteBox(w, bbOut);
                    w.WriteString("stl", stl);
                    w.WriteNumber("triangles", tris);
                    w.WriteString("mask", cfg.WriteMask ? stem + "_mask.json" : "");
                    w.WriteString("mask_raw", cfg.WriteRawMask ? stem + "_mask_raw.json" : "");
                    w.WritePropertyName("field");
                    w.WriteRawValue(field.DescribeJson());
                    w.WritePropertyName("timings_s");
                    w.WriteStartObject();
                    w.WriteNumber("tables", tTables);
                    w.WriteNumber("render", tRender);
                    w.WriteNumber("mask_raw", tMaskRaw);
                    w.WriteNumber("opening", tOpen);
                    w.WriteNumber("mask", tMask);
                    w.WriteNumber("stl", tStl);
                    w.WriteNumber("total", total.Elapsed.TotalSeconds);
                    w.WriteEndObject();
                    w.WriteEndObject();
                }
                File.WriteAllBytes(report, ms.ToArray());
            }
            Log($"done in {total.Elapsed.TotalSeconds:F1} s; report {report}");
            return report;
        }

        private static string SafeVersion()
        {
            try
            {
                return Library.strVersion();
            }
            catch (Exception)
            {
                return "unknown";
            }
        }

        private static void WriteBox(Utf8JsonWriter w, BBox3 b)
        {
            w.WriteStartArray();
            w.WriteNumberValue(b.vecMin.X);
            w.WriteNumberValue(b.vecMin.Y);
            w.WriteNumberValue(b.vecMin.Z);
            w.WriteNumberValue(b.vecMax.X);
            w.WriteNumberValue(b.vecMax.Y);
            w.WriteNumberValue(b.vecMax.Z);
            w.WriteEndArray();
        }

        private static string FrameJson(JacketField f, double renderHalf)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("{\"mode\": ").Append(MaskIO.Q(f.Sector ? "sector" : "full"));
            sb.Append(", \"sector_center_deg\": ").Append(ExportConfig.F(f.SectorCenter * 180.0 / Math.PI));
            if (renderHalf >= 0.0)
                sb.Append(", \"half_angle_deg\": ").Append(ExportConfig.F(renderHalf * 180.0 / Math.PI));
            else
                sb.Append(", \"half_angle_deg\": null");
            sb.Append(", \"nominal_half_angle_deg\": ").Append(ExportConfig.F(f.SectorHalf * 180.0 / Math.PI));
            sb.Append('}');
            return sb.ToString();
        }

        /// <summary>Write the voxel field as a bit-packed mask (solid = non-positive signed distance) from PicoGK slices.</summary>
        public static void WriteMask(Voxels vox, string stem, double h, string frameJson, string kind, JacketField f)
        {
            vox.GetVoxelDimensions(out int x0, out int y0, out int z0, out int nx, out int ny, out int nz);
            Vector3 org = Library.vecVoxelsToMm(x0, y0, z0);
            ImageGrayScale img = new ImageGrayScale(nx, ny);
            string meta = "{\"design\": " + MaskIO.Q(f.Fields.Label) + ", \"voxel_origin_index\": [" + x0 + ", " + y0 + ", " + z0 + "]}";
            MaskIO.Write(stem, nx, ny, nz, org.X, org.Y, org.Z, h, frameJson, "picogk", kind, meta,
                         (k, bits) =>
                         {
                             // PicoGK's native GetSlice fills the image from y = max down to y = min
                             // (row 0 = top), x ascending; the mask stores y ascending.
                             vox.GetVoxelSlice(k, ref img);
                             float[] v = img.m_afValues;
                             for (int row = 0; row < ny; row++)
                             {
                                 int src = row * nx;
                                 int dst = (ny - 1 - row) * nx;
                                 for (int x = 0; x < nx; x++)
                                     bits[dst + x] = v[src + x] <= 0f;
                             }
                         });
            Log($"mask {stem}.json: {nx} x {ny} x {nz} voxels, origin index ({x0}, {y0}, {z0})");
        }
    }
}
