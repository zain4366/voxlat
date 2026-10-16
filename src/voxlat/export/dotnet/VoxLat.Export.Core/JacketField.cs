// VoxLat Task 12 - signed distance of the graded TPMS cooling jacket, evaluated per voxel.
//
// This is a line-by-line mirror of voxlat/export/field.py (JacketField), which documents the geometry:
//   * phase-field grading in conformal cylindrical coordinates: integer number of cells around
//     (u_t = N P / T), axial phase from integrating 1 / (a_z L) (origin at mid-length), log-polar radial
//     phase u_r = phase_r0 + (r_m / L) ln(r / r_i), so cells never tear;
//   * blended TPMS phi = (1 - w) G / 1.5 + w D / sqrt 2, network solid phi <= c(w, rho) (Task 1 table);
//   * sd = (phi - c) / |grad phi| (floored), clipped to the gap minus the manifold slots, unioned with
//     sleeve, outer wall, end caps and port bosses, minus port bores; optional sector wedge.
// Units: mm and radians. Negative = solid (PicoGK convention). Thread-safe after construction.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace VoxLat.Export
{
    public sealed class JacketField
    {
        private const double TwoPi = 2.0 * Math.PI;
        private const double GAmp = 1.5;
        private static readonly double DAmp = Math.Sqrt(2.0);
        private const double GradFloor = 0.25;

        public readonly ExportConfig Config;
        public readonly DesignFields Fields;

        // geometry (mm)
        private readonly double m_rs, m_rli, m_rlo, m_ro, m_rm, m_Lax, m_zlo, m_zhi, m_tcap, m_delta, m_clamp;
        private readonly bool m_slots;
        private readonly double[] m_slotAng, m_slotHalf, m_slotZ0, m_slotZ1;
        private readonly double[] m_portCa, m_portSa, m_portZ, m_portRb, m_portRB, m_portTop;
        private readonly double m_tBore0, m_tBoss0;
        private readonly double m_phaseR0;
        private readonly bool m_conformal;
        private readonly double m_seam;
        private readonly bool m_rib;
        private readonly double m_ribT;

        // tables on (theta' = theta - seam, z): index [i * nz + j], i = 0 .. nt (nt + 1 rows), j = 0 .. nz - 1
        private readonly int m_nt, m_nz;
        private readonly double m_dth, m_dz;
        private readonly double[] m_Ut, m_Uz, m_iht, m_ihr, m_ihz, m_c, m_w, m_rho, m_L;

        // threshold table
        private readonly double[] m_tw, m_tr;
        private readonly double[,] m_tc;
        private readonly double m_tdw, m_tdr;

        // render frame
        public readonly bool Sector;
        public readonly double SectorCenter;   // rad
        public readonly double SectorHalf;     // rad (nominal)
        public readonly double SectorHalfWide; // rad (nominal + margin / r_i)
        private readonly double m_cc, m_cs;

        // early-out radii (exact: beyond them every primitive is >= clamp)
        private readonly double m_rFar, m_rNear, m_zFarLo, m_zFarHi;

        // diagnostics
        public readonly int NTheta;
        public readonly double CellsAroundMin, CellsAroundMax, IntegerDeviation;
        public readonly string CircumferentialMode;

        public JacketField(ExportConfig cfg, string? designPath = null)
        {
            Config = cfg;
            Fields = new DesignFields(designPath ?? cfg.DesignPath);
            m_rs = cfg.RStator;
            m_rli = cfg.RLatticeIn;
            m_rlo = cfg.RLatticeOut;
            m_ro = cfg.ROuter;
            m_rm = cfg.RMean;
            m_Lax = cfg.AxialLength;
            m_zlo = cfg.ZMin;
            m_zhi = cfg.ZMax;
            m_tcap = cfg.EndCap;
            m_delta = cfg.LatticeOverlap;
            m_clamp = cfg.SdClamp;
            if (Math.Abs(Fields.RMeanMm - m_rm) > 1e-6 || Math.Abs(Fields.AxialLengthMm - m_Lax) > 1e-6)
                throw new InvalidOperationException("design basis (r_m, L_ax) does not match the export geometry");
            m_slots = cfg.Layout == "slots";
            m_phaseR0 = cfg.PhaseR0;
            m_conformal = cfg.Conformal;
            m_seam = Mod(cfg.SeamAngleDeg * Math.PI / 180.0, TwoPi);
            m_ribT = cfg.SeamRibMm;

            // ---- slots (a slot reaching an axial end also removes the lattice overlap into the cap)
            List<double> sa = new List<double>(), sh = new List<double>(), s0 = new List<double>(), s1 = new List<double>();
            if (cfg.Slot != null)
            {
                SlotSpec s = cfg.Slot;
                double half = 0.5 * s.WidthMm / s.RRefMm;
                double z0 = s.Z0Mm, z1 = s.Z1Mm;
                if (z0 <= 1e-9) z0 = -2.0 * m_delta;
                if (z1 >= m_Lax - 1e-9) z1 = m_Lax + 2.0 * m_delta;
                foreach (double ang in new[] { s.InletAngleDeg, s.OutletAngleDeg })
                {
                    sa.Add(ang * Math.PI / 180.0);
                    sh.Add(half);
                    s0.Add(z0);
                    s1.Add(z1);
                }
            }
            m_slotAng = sa.ToArray();
            m_slotHalf = sh.ToArray();
            m_slotZ0 = s0.ToArray();
            m_slotZ1 = s1.ToArray();

            // ---- ports
            int np = cfg.Ports.Count;
            m_portCa = new double[np];
            m_portSa = new double[np];
            m_portZ = new double[np];
            m_portRb = new double[np];
            m_portRB = new double[np];
            m_portTop = new double[np];
            double topMax = m_ro, bossMax = 0.0, zpLo = m_zlo, zpHi = m_zhi;
            for (int k = 0; k < np; k++)
            {
                PortSpec p = cfg.Ports[k];
                double a = p.AngleDeg * Math.PI / 180.0;
                m_portCa[k] = Math.Cos(a);
                m_portSa[k] = Math.Sin(a);
                m_portZ[k] = p.ZMm;
                m_portRb[k] = p.BoreRadiusMm;
                m_portRB[k] = p.BossRadiusMm;
                m_portTop[k] = m_ro + p.BossHeightMm;
                topMax = Math.Max(topMax, m_portTop[k] + 1.0);
                bossMax = Math.Max(bossMax, Math.Max(p.BossRadiusMm, p.BoreRadiusMm));
                zpLo = Math.Min(zpLo, p.ZMm - Math.Max(p.BossRadiusMm, p.BoreRadiusMm));
                zpHi = Math.Max(zpHi, p.ZMm + Math.Max(p.BossRadiusMm, p.BoreRadiusMm));
            }
            m_tBore0 = 0.5 * (m_rli + m_rlo);
            m_tBoss0 = 0.5 * (m_rlo + m_ro);
            double c = m_clamp;
            m_rFar = Math.Sqrt((topMax + c) * (topMax + c) + (bossMax + c) * (bossMax + c)) + 1.0;
            m_rNear = m_rs - c - 1.0;
            m_zFarLo = Math.Min(zpLo, m_zlo) - c - 1.0;
            m_zFarHi = Math.Max(zpHi, m_zhi) + c + 1.0;

            // ---- threshold table
            m_tw = cfg.TableW;
            m_tr = cfg.TableRho;
            m_tc = cfg.TableC;
            m_tdw = (m_tw[m_tw.Length - 1] - m_tw[0]) / (m_tw.Length - 1);
            m_tdr = (m_tr[m_tr.Length - 1] - m_tr[0]) / (m_tr.Length - 1);

            // ---- phase tables
            m_nt = cfg.TableNTheta;
            m_nz = cfg.TableNZ;
            m_dth = TwoPi / m_nt;
            m_dz = m_Lax / (m_nz - 1);
            int nt = m_nt, nz = m_nz;
            double[] th = new double[nt + 1];
            double[] thAbs = new double[nt + 1];
            for (int i = 0; i <= nt; i++)
            {
                th[i] = i * m_dth;
                thAbs[i] = th[i] + m_seam;
            }
            double[] zz = new double[nz];
            for (int j = 0; j < nz; j++)
                zz[j] = j * m_dz;
            double[,] Frho = Fields.EvaluateGrid("rho", thAbs, zz);
            double[,] Fw = Fields.EvaluateGrid("w", thAbs, zz);
            double[,] FL = Fields.EvaluateGrid("L", thAbs, zz);
            double[,] Faz = Fields.EvaluateGrid("a_z", thAbs, zz);
            for (int j = 0; j < nz; j++)
            {
                Frho[nt, j] = Frho[0, j];
                Fw[nt, j] = Fw[0, j];
                FL[nt, j] = FL[0, j];
                Faz[nt, j] = Faz[0, j];
            }
            for (int i = 0; i <= nt; i++)
                for (int j = 0; j < nz; j++)
                    if (!(FL[i, j] > 0.0))
                        throw new InvalidOperationException("cell size field must be positive");

            // circumferential cumulative cell count P (trapezoid, cells at r_m)
            double[,] P = new double[nt + 1, nz];
            for (int j = 0; j < nz; j++)
            {
                double acc = 0.0;
                P[0, j] = 0.0;
                for (int i = 0; i < nt; i++)
                {
                    acc += 0.5 * m_dth * (m_rm / FL[i, j] + m_rm / FL[i + 1, j]);
                    P[i + 1, j] = acc;
                }
            }
            double[] Tz = new double[nz];
            double sumT = 0.0, tMin = double.MaxValue, tMax = double.MinValue;
            for (int j = 0; j < nz; j++)
            {
                Tz[j] = P[nt, j];
                sumT += Tz[j];
                tMin = Math.Min(tMin, Tz[j]);
                tMax = Math.Max(tMax, Tz[j]);
            }
            int N = cfg.NThetaCells ?? Math.Max(1, (int)Math.Round(sumT / nz, MidpointRounding.ToEven));
            if (N < 1)
                throw new InvalidOperationException("n_theta_cells must be >= 1");
            NTheta = N;
            CellsAroundMin = tMin;
            CellsAroundMax = tMax;
            double dev = 0.0;
            for (int j = 0; j < nz; j++)
                dev = Math.Max(dev, Math.Abs(Tz[j] / N - 1.0));
            IntegerDeviation = dev;
            string mode = cfg.Circumferential;
            if (mode != "integer" && mode != "seam")
                throw new InvalidOperationException($"circumferential must be 'integer' or 'seam', got '{mode}'");
            CircumferentialMode = mode;
            m_rib = false;
            if (mode == "seam")
            {
                bool hidden = false;
                for (int k = 0; k < m_slotAng.Length; k++)
                {
                    double dth = Math.Abs(Mod(m_seam - m_slotAng[k] + Math.PI, TwoPi) - Math.PI);
                    if (dth < m_slotHalf[k] - 1e-9 && m_slotZ0[k] <= 0.0 && m_slotZ1[k] >= m_Lax)
                        hidden = true;
                }
                m_rib = !hidden;
            }

            // axial phase: trapezoid of 1 / (a_z L), origin z_ref
            double[,] Uz = new double[nt + 1, nz];
            double[,] g = new double[nt + 1, nz];
            for (int i = 0; i <= nt; i++)
                for (int j = 0; j < nz; j++)
                    g[i, j] = 1.0 / (Faz[i, j] * FL[i, j]);
            for (int i = 0; i <= nt; i++)
            {
                double acc = 0.0;
                Uz[i, 0] = 0.0;
                for (int j = 0; j < nz - 1; j++)
                {
                    acc += 0.5 * m_dz * (g[i, j] + g[i, j + 1]);
                    Uz[i, j + 1] = acc;
                }
            }
            double tzr = Math.Min(Math.Max(cfg.AxialPhaseOrigin / m_dz, 0.0), nz - 1.0);
            int jr = Math.Min((int)Math.Floor(tzr), nz - 2);
            double br = tzr - jr;
            for (int i = 0; i <= nt; i++)
            {
                double sub = (1.0 - br) * Uz[i, jr] + br * Uz[i, jr + 1];
                for (int j = 0; j < nz; j++)
                    Uz[i, j] -= sub;
            }

            int n = (nt + 1) * nz;
            m_Ut = new double[n];
            m_Uz = new double[n];
            m_iht = new double[n];
            m_ihr = new double[n];
            m_ihz = new double[n];
            m_c = new double[n];
            m_w = new double[n];
            m_rho = new double[n];
            m_L = new double[n];
            for (int i = 0; i <= nt; i++)
            {
                for (int j = 0; j < nz; j++)
                {
                    int q = i * nz + j;
                    if (mode == "integer")
                    {
                        m_Ut[q] = N * P[i, j] / Tz[j] + cfg.PhaseTheta0;
                        m_iht[q] = N / (Tz[j] * FL[i, j]);
                    }
                    else
                    {
                        m_Ut[q] = P[i, j] + cfg.PhaseTheta0;
                        m_iht[q] = 1.0 / FL[i, j];
                    }
                    m_Uz[q] = Uz[i, j] + cfg.PhaseZ0;
                    m_ihr[q] = 1.0 / FL[i, j];
                    m_ihz[q] = g[i, j];
                    m_c[q] = Threshold(Fw[i, j], Frho[i, j]);
                    m_w[q] = Fw[i, j];
                    m_rho[q] = Frho[i, j];
                    m_L[q] = FL[i, j];
                }
            }

            // ---- render frame
            Sector = cfg.Mode == "sector";
            SectorCenter = cfg.SectorCenterDeg * Math.PI / 180.0;
            SectorHalf = 0.5 * cfg.SectorAngleDeg * Math.PI / 180.0;
            SectorHalfWide = SectorHalf + cfg.SectorMarginMm / m_rli;
            m_cc = Math.Cos(SectorCenter);
            m_cs = Math.Sin(SectorCenter);
        }

        public static JacketField Load(string configPath, string? designPath = null)
            => new JacketField(ExportConfig.Load(configPath), designPath);

        // ------------------------------------------------------------------ helpers
        private static double Mod(double x, double y)
        {
            double m = x % y;
            return m < 0.0 ? m + y : m;
        }

        /// <summary>Continuum TPMS threshold c(w, rho): bilinear on the Task 1 table, clipped (as Python).</summary>
        public double Threshold(double w, double rho)
        {
            double wc = Math.Min(Math.Max(w, m_tw[0]), m_tw[m_tw.Length - 1]);
            double rc = Math.Min(Math.Max(rho, m_tr[0]), m_tr[m_tr.Length - 1]);
            double fw = (wc - m_tw[0]) / m_tdw;
            double fr = (rc - m_tr[0]) / m_tdr;
            int i = Math.Min((int)Math.Floor(fw), m_tw.Length - 2);
            int j = Math.Min((int)Math.Floor(fr), m_tr.Length - 2);
            double a = fw - i, b = fr - j;
            return (1 - a) * (1 - b) * m_tc[i, j] + a * (1 - b) * m_tc[i + 1, j]
                 + (1 - a) * b * m_tc[i, j + 1] + a * b * m_tc[i + 1, j + 1];
        }

        private void Index(double theta, double z, out int q00, out double a, out double b)
        {
            double tp = theta - m_seam;
            if (tp < 0.0)
                tp += TwoPi;
            double tt = tp / m_dth;
            int i = (int)Math.Floor(tt);
            if (i < 0) i = 0;
            if (i > m_nt - 1) i = m_nt - 1;
            a = tt - i;
            double zc = Math.Min(Math.Max(z, 0.0), m_Lax);
            double tz = zc / m_dz;
            int j = Math.Min((int)Math.Floor(tz), m_nz - 2);
            b = tz - j;
            q00 = i * m_nz + j;
        }

        private double Interp(double[] T, int q00, double a, double b)
        {
            int q10 = q00 + m_nz;
            return (1 - a) * (1 - b) * T[q00] + a * (1 - b) * T[q10]
                 + (1 - a) * b * T[q00 + 1] + a * b * T[q10 + 1];
        }

        /// <summary>Tabulated fields at world angle theta (rad, [0, 2 pi)) and z (mm): rho, w, L, c.</summary>
        public void FieldsAt(double theta, double z, out double rho, out double w, out double L, out double c)
        {
            Index(Mod(theta, TwoPi), z, out int q, out double a, out double b);
            rho = Interp(m_rho, q, a, b);
            w = Interp(m_w, q, a, b);
            L = Interp(m_L, q, a, b);
            c = Interp(m_c, q, a, b);
        }

        // ------------------------------------------------------------------ the bare lattice
        /// <summary>First-order signed distance of the unclipped graded TPMS (world frame, mm).</summary>
        public double LatticeSd(double x, double y, double z, double r, double th)
        {
            Index(th, z, out int q, out double a, out double b);
            double ut = Interp(m_Ut, q, a, b);
            double uz = Interp(m_Uz, q, a, b);
            double iht = Interp(m_iht, q, a, b);
            double ihr = Interp(m_ihr, q, a, b);
            double ihz = Interp(m_ihz, q, a, b);
            double c = Interp(m_c, q, a, b);
            double w = Interp(m_w, q, a, b);
            double rs = Math.Max(r, 1e-9);
            double st = iht * m_rm / rs;
            double ur, sr;
            if (m_conformal)
            {
                ur = m_phaseR0 + ihr * m_rm * Math.Log(rs / m_rli);
                sr = ihr * m_rm / rs;
            }
            else
            {
                ur = m_phaseR0 + ihr * (r - m_rli);
                sr = ihr;
            }
            double X = TwoPi * ur, Y = TwoPi * ut, Z = TwoPi * uz;
            double sx = Math.Sin(X), cx = Math.Cos(X);
            double sy = Math.Sin(Y), cy = Math.Cos(Y);
            double sz = Math.Sin(Z), cz = Math.Cos(Z);
            double G = sx * cy + sy * cz + sz * cx;
            double Gx = cx * cy - sz * sx;
            double Gy = -sx * sy + cy * cz;
            double Gz = -sy * sz + cz * cx;
            double D = sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz;
            double Dx = cx * sy * sz + cx * cy * cz - sx * sy * cz - sx * cy * sz;
            double Dy = sx * cy * sz - sx * sy * cz + cx * cy * cz - cx * sy * sz;
            double Dz = sx * sy * cz - sx * cy * sz - cx * sy * sz + cx * cy * cz;
            double wg = (1.0 - w) / GAmp, wd = w / DAmp;
            double phi = wg * G + wd * D;
            double gx = (wg * Gx + wd * Dx) * TwoPi * sr;
            double gy = (wg * Gy + wd * Dy) * TwoPi * st;
            double gz = (wg * Gz + wd * Dz) * TwoPi * ihz;
            double gn = Math.Sqrt(gx * gx + gy * gy + gz * gz);
            double floor = GradFloor * TwoPi * (sr + st + ihz) / 3.0;
            return (phi - c) / Math.Max(gn, floor);
        }

        // ------------------------------------------------------------------ pieces
        private double LatticeRegion(double r, double th, double z)
        {
            double d = m_delta;
            double reg = Math.Max(m_rli - d - r, r - m_rlo - d);
            if (m_slots)
            {
                reg = Math.Max(reg, Math.Max(-d - z, z - m_Lax - d));
                for (int k = 0; k < m_slotAng.Length; k++)
                {
                    double dth = Math.Abs(Mod(th - m_slotAng[k] + Math.PI, TwoPi) - Math.PI);
                    double slotSd = Math.Max(r * (dth - m_slotHalf[k]), Math.Max(m_slotZ0[k] - z, z - m_slotZ1[k]));
                    reg = Math.Max(reg, -slotSd);
                }
            }
            else
            {
                reg = Math.Max(reg, Math.Max(-z, z - m_Lax));
            }
            return reg;
        }

        private double Housing(double r, double z)
        {
            double zb = Math.Max(m_zlo - z, z - m_zhi);
            double sleeve = Math.Max(Math.Max(m_rs - r, r - m_rli), zb);
            double outer = Math.Max(Math.Max(m_rlo - r, r - m_ro), zb);
            double rr = Math.Max(m_rs - r, r - m_ro);
            double capLo = Math.Max(rr, Math.Max(m_zlo - z, z - (m_zlo + m_tcap)));
            double capHi = Math.Max(rr, Math.Max((m_zhi - m_tcap) - z, z - m_zhi));
            return Math.Min(Math.Min(sleeve, outer), Math.Min(capLo, capHi));
        }

        private double Rib(double r, double th, double z)
        {
            double d = m_delta;
            double dth = Math.Abs(Mod(th - m_seam + Math.PI, TwoPi) - Math.PI);
            double rib = Math.Max(r * dth - 0.5 * m_ribT, Math.Max(m_rli - d - r, r - m_rlo - d));
            if (m_slots)
                return Math.Max(rib, Math.Max(-d - z, z - m_Lax - d));
            return Math.Max(rib, Math.Max(-z, z - m_Lax));
        }

        public double Wedge(double xl, double yl, double half)
        {
            double sb = Math.Sin(half), cb = Math.Cos(half);
            return Math.Max(-xl * sb + yl * cb, -xl * sb - yl * cb);
        }

        /// <summary>Signed distance (mm) at a point of the render frame (sector-local frame in sector mode).
        /// halfAngle: wedge half angle (rad); NaN = nominal sector in sector mode / none in full mode;
        /// a negative value = no wedge.</summary>
        public double SignedDistance(double xl, double yl, double z, double halfAngle)
        {
            double clamp = m_clamp;
            double half = double.IsNaN(halfAngle) ? (Sector ? SectorHalf : -1.0) : halfAngle;
            double wedge = double.NegativeInfinity;
            if (half >= 0.0)
            {
                wedge = Wedge(xl, yl, half);
                if (wedge >= clamp)
                    return clamp;
            }
            double x = xl, y = yl;
            if (Sector)
            {
                x = m_cc * xl - m_cs * yl;
                y = m_cs * xl + m_cc * yl;
            }
            double r = Math.Sqrt(x * x + y * y);
            if (r > m_rFar || r < m_rNear || z < m_zFarLo || z > m_zFarHi)
                return clamp; // every primitive is >= clamp here (exact)
            double th = Math.Atan2(y, x);
            if (th < 0.0)
                th += TwoPi;

            double reg = LatticeRegion(r, th, z);
            double lat = reg;
            if (reg < clamp)
                lat = Math.Max(LatticeSd(x, y, z, r, th), reg);
            double solid = Math.Min(Housing(r, z), lat);
            if (m_rib)
                solid = Math.Min(solid, Rib(r, th, z));
            int np = m_portCa.Length;
            // bosses first (union), then bores (difference) - same order as Python
            Span<double> bores = np <= 8 ? stackalloc double[np] : new double[np];
            for (int k = 0; k < np; k++)
            {
                double ca = m_portCa[k], sa = m_portSa[k];
                double dz = z - m_portZ[k];
                double t = x * ca + y * sa;
                double px = x - t * ca, py = y - t * sa;
                double dperp = Math.Sqrt(px * px + py * py + dz * dz);
                double top = m_portTop[k];
                bores[k] = Math.Max(dperp - m_portRb[k], Math.Max(m_tBore0 - t, t - (top + 1.0)));
                double boss = Math.Max(dperp - m_portRB[k], Math.Max(m_tBoss0 - t, t - top));
                solid = Math.Min(solid, boss);
            }
            for (int k = 0; k < np; k++)
                solid = Math.Max(solid, -bores[k]);
            if (half >= 0.0)
                solid = Math.Max(solid, wedge);
            if (solid > clamp) return clamp;
            if (solid < -clamp) return -clamp;
            return solid;
        }

        /// <summary>Distance (mm) to the nearest open lattice face (slot sides / ring faces) at a point of the
        /// render frame - where clipping leaves slivers (mirror of JacketField.interface_distance in Python).</summary>
        public double InterfaceDistance(double xl, double yl, double z)
        {
            double x = xl, y = yl;
            if (Sector)
            {
                x = m_cc * xl - m_cs * yl;
                y = m_cs * xl + m_cc * yl;
            }
            if (!m_slots)
                return Math.Min(Math.Abs(z), Math.Abs(z - m_Lax));
            double r = Math.Sqrt(x * x + y * y);
            double th = Math.Atan2(y, x);
            if (th < 0.0)
                th += TwoPi;
            double d = double.PositiveInfinity;
            for (int k = 0; k < m_slotAng.Length; k++)
            {
                double dth = Math.Abs(Mod(th - m_slotAng[k] + Math.PI, TwoPi) - Math.PI);
                d = Math.Min(d, Math.Abs(r * (dth - m_slotHalf[k])));
            }
            return d;
        }

        // ------------------------------------------------------------------ bounding box (render frame)
        /// <summary>Axis-aligned render box (mm), mirror of voxlat.export.sector.render_bbox.</summary>
        public void RenderBox(double halfAngle, out double[] lo, out double[] hi)
        {
            double top = m_ro;
            foreach (PortSpec p in Config.Ports)
                top = Math.Max(top, m_ro + p.BossHeightMm);
            top += 1.0;
            const double pad = 0.5;
            if (Sector)
            {
                double b = halfAngle;
                lo = new[] { m_rs * Math.Cos(b) - pad, -top * Math.Sin(b) - pad, m_zlo - pad };
                hi = new[] { top + pad, top * Math.Sin(b) + pad, m_zhi + pad };
            }
            else
            {
                lo = new[] { -top - pad, -top - pad, m_zlo - pad };
                hi = new[] { top + pad, top + pad, m_zhi + pad };
            }
        }

        public string DescribeJson()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append('{');
            sb.Append("\"circumferential_mode\":\"").Append(CircumferentialMode).Append("\",");
            sb.Append("\"N_theta\":").Append(NTheta.ToString(CultureInfo.InvariantCulture)).Append(',');
            sb.Append("\"cells_around_min\":").Append(ExportConfig.F(CellsAroundMin)).Append(',');
            sb.Append("\"cells_around_max\":").Append(ExportConfig.F(CellsAroundMax)).Append(',');
            sb.Append("\"integer_N_max_size_deviation\":").Append(ExportConfig.F(IntegerDeviation)).Append(',');
            sb.Append("\"seam_rib\":").Append(m_rib ? "true" : "false").Append(',');
            sb.Append("\"table\":[").Append(m_nt).Append(',').Append(m_nz).Append(']');
            sb.Append('}');
            return sb.ToString();
        }
    }
}
