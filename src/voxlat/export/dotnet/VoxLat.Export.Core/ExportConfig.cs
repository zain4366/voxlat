// VoxLat Task 12 - export configuration (mirror of voxlat/export/config.py, schema voxlat-export-config/1).
// All lengths in mm, angles in degrees as stored in the JSON; converted to radians where used.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace VoxLat.Export
{
    public sealed class SlotSpec
    {
        public double InletAngleDeg;
        public double OutletAngleDeg;
        public double WidthMm;
        public double RRefMm;
        public double Z0Mm;
        public double Z1Mm;
    }

    public sealed class PortSpec
    {
        public string Name = "";
        public double AngleDeg;
        public double ZMm;
        public double BoreRadiusMm;
        public double BossRadiusMm;
        public double BossHeightMm;
    }

    public sealed class ExportConfig
    {
        public const string Schema = "voxlat-export-config/1";

        public string ConfigPath = "";
        public string DesignPath = "";
        public string DesignLabel = "";
        public string Layout = "slots";

        // geometry
        public double RStator, RLatticeIn, RLatticeOut, ROuter, RMean, AxialLength, ZMin, ZMax, EndCap, LatticeOverlap;
        public SlotSpec? Slot;
        public double? RingHeight;
        public List<PortSpec> Ports = new List<PortSpec>();

        // lattice
        public int? NThetaCells;
        public double PhaseR0, PhaseTheta0, PhaseZ0, AxialPhaseOrigin;
        public bool Conformal = true;
        public string Circumferential = "integer";
        public double SeamAngleDeg;
        public double SeamRibMm;
        public int TableNTheta = 2048;
        public int TableNZ = 257;
        public double SdClamp = 2.0;

        // threshold table c(w, rho)
        public double[] TableW = Array.Empty<double>();
        public double[] TableRho = Array.Empty<double>();
        public double[,] TableC = new double[0, 0];

        // render
        public string Mode = "sector";
        public double VoxelSize;
        public double SectorCenterDeg;
        public double SectorAngleDeg;
        public double SectorMarginMm;
        public double SliverRadiusMm;
        public double SliverBandMm;
        public bool SaveStl = true;
        public bool WriteMask = true;
        public bool WriteRawMask = true;
        public bool ShowInViewer = true;
        public string OutputDir = ".";
        public string Name = "jacket";

        /// <summary>Parse JSON text; bare NaN / Infinity tokens (Python json.dumps) are turned into null.</summary>
        public static JsonDocument ParseJson(string text)
        {
            string clean = Regex.Replace(text, @"(?<=[:\[,\s])-?(NaN|Infinity)(?=[\s,\]\}])", "null");
            return JsonDocument.Parse(clean, new JsonDocumentOptions { AllowTrailingCommas = true });
        }

        public static ExportConfig Load(string path)
        {
            string text = File.ReadAllText(path);
            using JsonDocument doc = ParseJson(text);
            JsonElement root = doc.RootElement;
            string schema = root.GetProperty("schema").GetString() ?? "";
            if (schema != Schema)
                throw new InvalidDataException($"unsupported export config schema '{schema}' (expected {Schema})");

            ExportConfig c = new ExportConfig();
            c.ConfigPath = Path.GetFullPath(path);
            c.DesignPath = root.GetProperty("design_path").GetString() ?? "";
            if (root.TryGetProperty("design_label", out JsonElement lbl) && lbl.ValueKind == JsonValueKind.String)
                c.DesignLabel = lbl.GetString() ?? "";
            c.Layout = root.GetProperty("layout").GetString() ?? "slots";
            if (c.Layout != "slots" && c.Layout != "end_rings")
                throw new InvalidDataException($"unknown layout '{c.Layout}'");

            JsonElement g = root.GetProperty("geometry");
            c.RStator = g.GetProperty("r_stator_mm").GetDouble();
            c.RLatticeIn = g.GetProperty("r_lattice_in_mm").GetDouble();
            c.RLatticeOut = g.GetProperty("r_lattice_out_mm").GetDouble();
            c.ROuter = g.GetProperty("r_outer_mm").GetDouble();
            c.RMean = g.GetProperty("r_mean_mm").GetDouble();
            c.AxialLength = g.GetProperty("axial_length_mm").GetDouble();
            c.ZMin = g.GetProperty("z_min_mm").GetDouble();
            c.ZMax = g.GetProperty("z_max_mm").GetDouble();
            c.EndCap = g.GetProperty("end_cap_mm").GetDouble();
            c.LatticeOverlap = g.GetProperty("lattice_overlap_mm").GetDouble();
            JsonElement slot = g.GetProperty("slot");
            if (slot.ValueKind == JsonValueKind.Object)
            {
                c.Slot = new SlotSpec
                {
                    InletAngleDeg = slot.GetProperty("inlet_angle_deg").GetDouble(),
                    OutletAngleDeg = slot.GetProperty("outlet_angle_deg").GetDouble(),
                    WidthMm = slot.GetProperty("width_mm").GetDouble(),
                    RRefMm = slot.GetProperty("r_ref_mm").GetDouble(),
                    Z0Mm = slot.GetProperty("z0_mm").GetDouble(),
                    Z1Mm = slot.GetProperty("z1_mm").GetDouble(),
                };
            }
            JsonElement ring = g.GetProperty("ring");
            if (ring.ValueKind == JsonValueKind.Object)
                c.RingHeight = ring.GetProperty("height_mm").GetDouble();
            foreach (JsonElement p in g.GetProperty("ports").EnumerateArray())
            {
                c.Ports.Add(new PortSpec
                {
                    Name = p.GetProperty("name").GetString() ?? "",
                    AngleDeg = p.GetProperty("angle_deg").GetDouble(),
                    ZMm = p.GetProperty("z_mm").GetDouble(),
                    BoreRadiusMm = p.GetProperty("bore_radius_mm").GetDouble(),
                    BossRadiusMm = p.GetProperty("boss_radius_mm").GetDouble(),
                    BossHeightMm = p.GetProperty("boss_height_mm").GetDouble(),
                });
            }

            JsonElement lat = root.GetProperty("lattice");
            JsonElement nCells = lat.GetProperty("n_theta_cells");
            c.NThetaCells = nCells.ValueKind == JsonValueKind.Number ? nCells.GetInt32() : (int?)null;
            c.PhaseR0 = lat.GetProperty("phase_r0").GetDouble();
            c.PhaseTheta0 = lat.GetProperty("phase_theta0").GetDouble();
            c.PhaseZ0 = lat.GetProperty("phase_z0").GetDouble();
            c.AxialPhaseOrigin = lat.GetProperty("axial_phase_origin_mm").GetDouble();
            c.Conformal = lat.GetProperty("conformal").GetBoolean();
            c.Circumferential = lat.GetProperty("circumferential").GetString() ?? "integer";
            c.SeamAngleDeg = lat.GetProperty("seam_angle_deg").GetDouble();
            c.SeamRibMm = lat.GetProperty("seam_rib_mm").GetDouble();
            c.TableNTheta = lat.GetProperty("table_n_theta").GetInt32();
            c.TableNZ = lat.GetProperty("table_n_z").GetInt32();
            c.SdClamp = lat.GetProperty("sd_clamp_mm").GetDouble();
            if (c.TableNTheta < 8 || c.TableNZ < 2)
                throw new InvalidDataException("phase table too small");

            JsonElement tt = root.GetProperty("threshold_table");
            c.TableW = ReadVector(tt.GetProperty("w"));
            c.TableRho = ReadVector(tt.GetProperty("rho"));
            JsonElement cc = tt.GetProperty("c");
            int nw = cc.GetArrayLength();
            if (nw != c.TableW.Length)
                throw new InvalidDataException("threshold table: c rows != len(w)");
            c.TableC = new double[nw, c.TableRho.Length];
            int i = 0;
            foreach (JsonElement row in cc.EnumerateArray())
            {
                double[] v = ReadVector(row);
                if (v.Length != c.TableRho.Length)
                    throw new InvalidDataException("threshold table: c columns != len(rho)");
                for (int j = 0; j < v.Length; j++)
                    c.TableC[i, j] = v[j];
                i++;
            }

            JsonElement r = root.GetProperty("render");
            c.Mode = r.GetProperty("mode").GetString() ?? "sector";
            if (c.Mode != "sector" && c.Mode != "full")
                throw new InvalidDataException($"unknown render mode '{c.Mode}'");
            c.VoxelSize = r.GetProperty("voxel_size_mm").GetDouble();
            c.SectorCenterDeg = r.GetProperty("sector_center_deg").GetDouble();
            c.SectorAngleDeg = r.GetProperty("sector_angle_deg").GetDouble();
            c.SectorMarginMm = r.GetProperty("sector_margin_mm").GetDouble();
            c.SliverRadiusMm = r.GetProperty("sliver_radius_mm").GetDouble();
            c.SliverBandMm = r.GetProperty("sliver_band_mm").GetDouble();
            c.SaveStl = r.GetProperty("save_stl").GetBoolean();
            c.WriteMask = r.GetProperty("write_mask").GetBoolean();
            c.WriteRawMask = r.GetProperty("write_raw_mask").GetBoolean();
            c.ShowInViewer = r.GetProperty("show_in_viewer").GetBoolean();
            c.OutputDir = r.GetProperty("output_dir").GetString() ?? ".";
            c.Name = r.GetProperty("name").GetString() ?? "jacket";
            return c;
        }

        public static double[] ReadVector(JsonElement arr)
        {
            double[] v = new double[arr.GetArrayLength()];
            int k = 0;
            foreach (JsonElement e in arr.EnumerateArray())
                v[k++] = e.GetDouble();
            return v;
        }

        public static string F(double v) => v.ToString("R", CultureInfo.InvariantCulture);
    }
}
