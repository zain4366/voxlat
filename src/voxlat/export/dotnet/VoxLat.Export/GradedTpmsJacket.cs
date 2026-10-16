// VoxLat Task 12 - the graded TPMS jacket as a PicoGK implicit.
// All evaluation happens in C# (JacketField.SignedDistance); PicoGK calls fSignedDistance once per voxel
// of the bounding box during Voxels(IImplicit, BBox3) / RenderImplicit.

using System;
using System.Numerics;
using PicoGK;

namespace VoxLat.Export
{
    /// <summary>PicoGK implicit of the exported jacket (render frame, mm; negative = solid).</summary>
    public sealed class GradedTpmsJacket : IImplicit
    {
        public readonly JacketField Field;
        private readonly double m_half;
        private readonly BBox3 m_bounds;

        /// <param name="field">the jacket geometry</param>
        /// <param name="halfAngle">sector wedge half angle (rad); NaN = nominal sector (or none in full mode);
        /// negative = no wedge</param>
        public GradedTpmsJacket(JacketField field, double halfAngle)
        {
            Field = field;
            m_half = halfAngle;
            double boxHalf = double.IsNaN(halfAngle) || halfAngle < 0.0 ? field.SectorHalfWide : halfAngle;
            field.RenderBox(boxHalf, out double[] lo, out double[] hi);
            m_bounds = new BBox3(new Vector3((float)lo[0], (float)lo[1], (float)lo[2]),
                                 new Vector3((float)hi[0], (float)hi[1], (float)hi[2]));
        }

        public static GradedTpmsJacket FromFiles(string configPath, double halfAngle = double.NaN)
            => new GradedTpmsJacket(JacketField.Load(configPath), halfAngle);

        /// <summary>Render box (IBoundedImplicit only exists from PicoGK 1.7.6 on, so it is a plain property).</summary>
        public BBox3 oBounds => m_bounds;

        public float fSignedDistance(in Vector3 vec)
            => (float)Field.SignedDistance(vec.X, vec.Y, vec.Z, m_half);
    }

    /// <summary>Wedge of half angle around +x (sector-local frame), used to trim the sector STL.</summary>
    public sealed class WedgeImplicit : IImplicit
    {
        private readonly float m_sb, m_cb;

        public WedgeImplicit(double halfAngle)
        {
            m_sb = (float)Math.Sin(halfAngle);
            m_cb = (float)Math.Cos(halfAngle);
        }

        public float fSignedDistance(in Vector3 vec)
            => Math.Max(-vec.X * m_sb + vec.Y * m_cb, -vec.X * m_sb - vec.Y * m_cb);
    }

    /// <summary>Band of width band (mm) around the open lattice faces (inside = true) or its complement.</summary>
    public sealed class InterfaceBandImplicit : IImplicit
    {
        private readonly JacketField m_field;
        private readonly double m_band;
        private readonly bool m_inside;

        public InterfaceBandImplicit(JacketField field, double band, bool inside)
        {
            m_field = field;
            m_band = band;
            m_inside = inside;
        }

        public float fSignedDistance(in Vector3 vec)
        {
            double d = m_field.InterfaceDistance(vec.X, vec.Y, vec.Z) - m_band;
            if (d > 10.0) d = 10.0;
            return (float)(m_inside ? d : -d);
        }
    }
}
