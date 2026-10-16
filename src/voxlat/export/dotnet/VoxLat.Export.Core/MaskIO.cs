// VoxLat Task 12 - bit-packed voxel masks (format of voxlat/export/maskio.py, schema voxlat-mask/1) and the
// probe used to cross-check this C# implicit against the numpy mirror.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;

namespace VoxLat.Export
{
    public static class MaskIO
    {
        public const string Schema = "voxlat-mask/1";

        /// <summary>
        /// Write a mask: header stem.json + data stem.bin. sliceSolid(k, bits) must fill bits[x + y * nx] for
        /// slice k (true = solid). Slices are written in order z = 0 .. nz - 1, x fastest, little-endian bits.
        /// </summary>
        public static void Write(string stem, int nx, int ny, int nz, double ox, double oy, double oz, double voxel,
                                 string frameJson, string source, string kind, string metaJson,
                                 Action<int, bool[]> sliceSolid)
        {
            string? dir = Path.GetDirectoryName(Path.GetFullPath(stem));
            if (!string.IsNullOrEmpty(dir))
                Directory.CreateDirectory(dir);
            int bitsPerSlice = nx * ny;
            int bytesPerSlice = (bitsPerSlice + 7) / 8;
            bool[] bits = new bool[bitsPerSlice];
            byte[] buf = new byte[bytesPerSlice];
            using (FileStream fs = new FileStream(stem + ".bin", FileMode.Create, FileAccess.Write))
            {
                for (int k = 0; k < nz; k++)
                {
                    Array.Clear(bits, 0, bits.Length);
                    Array.Clear(buf, 0, buf.Length);
                    sliceSolid(k, bits);
                    for (int q = 0; q < bitsPerSlice; q++)
                        if (bits[q])
                            buf[q >> 3] |= (byte)(1 << (q & 7));
                    fs.Write(buf, 0, buf.Length);
                }
            }
            StringBuilder h = new StringBuilder();
            h.Append("{\n");
            h.Append(" \"schema\": \"").Append(Schema).Append("\",\n");
            h.Append(" \"shape\": [").Append(nx).Append(", ").Append(ny).Append(", ").Append(nz).Append("],\n");
            h.Append(" \"origin_mm\": [").Append(ExportConfig.F(ox)).Append(", ").Append(ExportConfig.F(oy)).Append(", ")
             .Append(ExportConfig.F(oz)).Append("],\n");
            h.Append(" \"voxel_mm\": ").Append(ExportConfig.F(voxel)).Append(",\n");
            h.Append(" \"frame\": ").Append(frameJson).Append(",\n");
            h.Append(" \"source\": \"").Append(source).Append("\",\n");
            h.Append(" \"kind\": \"").Append(kind).Append("\",\n");
            h.Append(" \"order\": \"slices along z; inside a slice x fastest then y; bit 1 = solid; little-endian bits\",\n");
            h.Append(" \"y_order\": \"ascending\",\n");
            h.Append(" \"meta\": ").Append(metaJson).Append("\n");
            h.Append("}\n");
            File.WriteAllText(stem + ".json", h.ToString());
        }

        public static string Q(string s) => "\"" + s.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";
    }

    /// <summary>Evaluate the implicit at many points (cross-check against numpy; no PicoGK needed).</summary>
    public static class Probe
    {
        /// <summary>
        /// pointsPath: float64 little-endian, N x 3 (x, y, z in the render frame, mm).
        /// Writes outPath: float64 N signed distances. halfAngleDeg: NaN = default wedge (see SignedDistance),
        /// negative = no wedge. Returns the JSON diagnostics of the field.
        /// </summary>
        public static string EvaluateFile(string configPath, string pointsPath, string outPath, double halfAngleDeg)
        {
            JacketField f = JacketField.Load(configPath);
            byte[] raw = File.ReadAllBytes(pointsPath);
            int n = raw.Length / 24;
            double[] pts = new double[n * 3];
            Buffer.BlockCopy(raw, 0, pts, 0, n * 24);
            double half = double.IsNaN(halfAngleDeg) ? double.NaN : (halfAngleDeg < 0 ? -1.0 : halfAngleDeg * Math.PI / 180.0);
            double[] sd = new double[n];
            for (int i = 0; i < n; i++)
                sd[i] = f.SignedDistance(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2], half);
            byte[] outBytes = new byte[n * 8];
            Buffer.BlockCopy(sd, 0, outBytes, 0, n * 8);
            File.WriteAllBytes(outPath, outBytes);
            return f.DescribeJson();
        }

        /// <summary>Tabulated fields (rho, w, L, c) at world (theta [rad], z [mm]) pairs: in N x 2, out N x 4.</summary>
        public static void FieldsFile(string configPath, string pointsPath, string outPath)
        {
            JacketField f = JacketField.Load(configPath);
            byte[] raw = File.ReadAllBytes(pointsPath);
            int n = raw.Length / 16;
            double[] pts = new double[n * 2];
            Buffer.BlockCopy(raw, 0, pts, 0, n * 16);
            double[] o = new double[n * 4];
            for (int i = 0; i < n; i++)
            {
                f.FieldsAt(pts[2 * i], pts[2 * i + 1], out double rho, out double w, out double L, out double c);
                o[4 * i] = rho;
                o[4 * i + 1] = w;
                o[4 * i + 2] = L;
                o[4 * i + 3] = c;
            }
            byte[] outBytes = new byte[n * 32];
            Buffer.BlockCopy(o, 0, outBytes, 0, n * 32);
            File.WriteAllBytes(outPath, outBytes);
        }

        /// <summary>Time n evaluations at random points of the render box (ns per call).</summary>
        public static double Benchmark(string configPath, int n)
        {
            JacketField f = JacketField.Load(configPath);
            f.RenderBox(f.Sector ? f.SectorHalfWide : 0.0, out double[] lo, out double[] hi);
            Random rng = new Random(1);
            double acc = 0.0;
            System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < n; i++)
            {
                double x = lo[0] + (hi[0] - lo[0]) * rng.NextDouble();
                double y = lo[1] + (hi[1] - lo[1]) * rng.NextDouble();
                double z = lo[2] + (hi[2] - lo[2]) * rng.NextDouble();
                acc += f.SignedDistance(x, y, z, double.NaN);
            }
            sw.Stop();
            if (double.IsNaN(acc))
                throw new InvalidOperationException("NaN signed distance");
            return sw.Elapsed.TotalMilliseconds * 1e6 / n;
        }
    }
}
