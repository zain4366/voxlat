// VoxLat Task 12 - B-spline design fields of a Task 11 design (mirror of DesignFields in voxlat/export/field.py
// and of voxlat/optimize/parameterization.py).
//
//   f(theta, z) = sum_ij P_ij Bs_i(theta / 2 pi) Bz_j(z / L_ax)
//   Bs: periodic uniform cubic B-spline, control i centred at theta_i = i 360 / n_s deg
//   Bz: clamped uniform B-spline of degree min(3, n_z - 1) on [0, 1] (Cox-de Boor, t = 1 -> last span)
// The control nets in the design JSON are physical (SI); L is converted to mm here.

using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace VoxLat.Export
{
    public sealed class DesignFields
    {
        public static readonly string[] Names = { "rho", "w", "L", "a_z" };

        public readonly double RMeanMm;
        public readonly double AxialLengthMm;
        public readonly string Layout;
        public readonly string Label;
        private readonly Dictionary<string, double[,]> m_nets = new Dictionary<string, double[,]>();
        private readonly Dictionary<string, double> m_const = new Dictionary<string, double>();

        public DesignFields(string designPath)
        {
            using JsonDocument doc = ExportConfig.ParseJson(File.ReadAllText(designPath));
            JsonElement root = doc.RootElement;
            Layout = root.GetProperty("layout").GetString() ?? "";
            Label = root.TryGetProperty("label", out JsonElement l) && l.ValueKind == JsonValueKind.String ? (l.GetString() ?? "") : "";
            JsonElement basis = root.GetProperty("basis");
            RMeanMm = basis.GetProperty("r_mean_m").GetDouble() * 1e3;
            AxialLengthMm = basis.GetProperty("axial_length_m").GetDouble() * 1e3;
            JsonElement nets = root.GetProperty("control_nets");
            Dictionary<string, double> fixedVals = new Dictionary<string, double>();
            if (root.TryGetProperty("fixed_fields", out JsonElement ff) && ff.ValueKind == JsonValueKind.Object)
            {
                foreach (JsonProperty p in ff.EnumerateObject())
                    if (p.Value.ValueKind == JsonValueKind.Number)
                        fixedVals[p.Name] = p.Value.GetDouble();
            }
            else if (basis.TryGetProperty("spec", out JsonElement spec) && spec.TryGetProperty("fixed", out JsonElement fx)
                     && fx.ValueKind == JsonValueKind.Object)
            {
                foreach (JsonProperty p in fx.EnumerateObject())
                    if (p.Value.ValueKind == JsonValueKind.Number)
                        fixedVals[p.Name] = p.Value.GetDouble();
            }
            foreach (string name in Names)
            {
                if (nets.TryGetProperty(name, out JsonElement net) && net.ValueKind == JsonValueKind.Array)
                {
                    int ns = net.GetArrayLength();
                    double[,]? P = null;
                    int i = 0;
                    foreach (JsonElement row in net.EnumerateArray())
                    {
                        double[] v = ExportConfig.ReadVector(row);
                        if (P == null)
                            P = new double[ns, v.Length];
                        if (v.Length != P.GetLength(1))
                            throw new InvalidDataException($"control net {name} is ragged");
                        for (int j = 0; j < v.Length; j++)
                            P[i, j] = v[j] * (name == "L" ? 1e3 : 1.0);
                        i++;
                    }
                    if (P == null || P.GetLength(1) < 1)
                        throw new InvalidDataException($"control net {name} is empty");
                    m_nets[name] = P;
                }
                else
                {
                    m_const[name] = fixedVals.TryGetValue(name, out double v) ? v : 1.0;
                }
            }
            foreach (string req in new[] { "rho", "w", "L" })
                if (!m_nets.ContainsKey(req))
                    throw new InvalidDataException($"design has no control net '{req}'");
        }

        private static double B3(double t)
        {
            double a = Math.Abs(t);
            if (a < 1.0)
                return (4.0 - 6.0 * a * a + 3.0 * a * a * a) / 6.0;
            if (a < 2.0)
                return (2.0 - a) * (2.0 - a) * (2.0 - a) / 6.0;
            return 0.0;
        }

        /// <summary>Periodic uniform cubic B-spline basis at u (period 1), n controls; B has length n.</summary>
        public static void PeriodicBasis(double u, int n, double[] B)
        {
            if (n == 1)
            {
                B[0] = 1.0;
                return;
            }
            double t = (u * n) % n;
            if (t < 0.0)
                t += n;
            for (int i = 0; i < n; i++)
            {
                double s = 0.0;
                for (int k = -2; k <= 2; k++)
                    s += B3(t - i - k * n);
                B[i] = s;
            }
        }

        /// <summary>Clamped uniform B-spline basis of degree min(3, n - 1) at t in [0, 1]; B has length n.</summary>
        public static void ClampedBasis(double t, int n, double[] B)
        {
            if (t < 0.0) t = 0.0;
            if (t > 1.0) t = 1.0;
            if (n == 1)
            {
                B[0] = 1.0;
                return;
            }
            int p = Math.Min(3, n - 1);
            int m = n - p;
            int nk = p + (m + 1) + p;
            double[] knots = new double[nk];
            for (int i = 0; i < p; i++)
                knots[i] = 0.0;
            for (int i = 0; i <= m; i++)
                knots[p + i] = (double)i / m;
            for (int i = 0; i < p; i++)
                knots[p + m + 1 + i] = 1.0;
            double[] cur = new double[nk - 1];
            int last = nk - p - 2;
            if (t >= 1.0)
            {
                cur[last] = 1.0;
            }
            else
            {
                for (int i = 0; i < nk - 1; i++)
                    if (knots[i] < knots[i + 1] && t >= knots[i] && t < knots[i + 1])
                        cur[i] = 1.0;
            }
            for (int k = 1; k <= p; k++)
            {
                double[] nxt = new double[nk - 1 - k];
                for (int i = 0; i < nk - 1 - k; i++)
                {
                    double d1 = knots[i + k] - knots[i];
                    double d2 = knots[i + k + 1] - knots[i + 1];
                    double a = d1 > 0.0 ? (t - knots[i]) / d1 * cur[i] : 0.0;
                    double b = d2 > 0.0 ? (knots[i + k + 1] - t) / d2 * cur[i + 1] : 0.0;
                    nxt[i] = a + b;
                }
                cur = nxt;
            }
            for (int i = 0; i < n; i++)
                B[i] = cur[i];
        }

        /// <summary>Field on the tensor grid theta[a] (rad) x z[b] (mm): result[a, b].</summary>
        public double[,] EvaluateGrid(string name, double[] theta, double[] z)
        {
            double[,] F = new double[theta.Length, z.Length];
            if (!m_nets.TryGetValue(name, out double[,]? P))
            {
                double v = m_const[name];
                for (int a = 0; a < theta.Length; a++)
                    for (int b = 0; b < z.Length; b++)
                        F[a, b] = v;
                return F;
            }
            int ns = P.GetLength(0), nz = P.GetLength(1);
            double[,] Bz = new double[z.Length, nz];
            double[] bz = new double[nz];
            for (int b = 0; b < z.Length; b++)
            {
                ClampedBasis(z[b] / AxialLengthMm, nz, bz);
                for (int j = 0; j < nz; j++)
                    Bz[b, j] = bz[j];
            }
            double[] bs = new double[ns];
            double[] row = new double[nz];
            for (int a = 0; a < theta.Length; a++)
            {
                PeriodicBasis(theta[a] / (2.0 * Math.PI), ns, bs);
                // row_j = sum_i Bs_i P_ij
                for (int j = 0; j < nz; j++)
                {
                    double s = 0.0;
                    for (int i = 0; i < ns; i++)
                        s += bs[i] * P[i, j];
                    row[j] = s;
                }
                for (int b = 0; b < z.Length; b++)
                {
                    double s = 0.0;
                    for (int j = 0; j < nz; j++)
                        s += row[j] * Bz[b, j];
                    F[a, b] = s;
                }
            }
            return F;
        }
    }
}
