// VoxLat Task 12 - command-line probe of the C# implicit (no PicoGK needed).
//   VoxLat.Export.Probe sd     <export_config.json> <points.bin (N x 3 float64)> <out.bin> [half_angle_deg | nan | -1]
//   VoxLat.Export.Probe fields <export_config.json> <theta_z.bin (N x 2 float64)> <out.bin (N x 4: rho, w, L, c)>
//   VoxLat.Export.Probe bench  <export_config.json> [n]
// Used by tests/test_export_dotnet.py and scripts/task12_export.py selftest.

using System;
using System.Globalization;

namespace VoxLat.Export
{
    public static class Program
    {
        public static int Main(string[] args)
        {
            try
            {
                if (args.Length >= 4 && args[0] == "sd")
                {
                    double half = double.NaN;
                    if (args.Length >= 5 && !args[4].Equals("nan", StringComparison.OrdinalIgnoreCase))
                        half = double.Parse(args[4], CultureInfo.InvariantCulture);
                    Console.WriteLine(Probe.EvaluateFile(args[1], args[2], args[3], half));
                    return 0;
                }
                if (args.Length >= 4 && args[0] == "fields")
                {
                    Probe.FieldsFile(args[1], args[2], args[3]);
                    return 0;
                }
                if (args.Length >= 2 && args[0] == "bench")
                {
                    int n = args.Length >= 3 ? int.Parse(args[2], CultureInfo.InvariantCulture) : 1000000;
                    double ns = Probe.Benchmark(args[1], n);
                    Console.WriteLine($"{{\"ns_per_eval\": {ns.ToString("F1", CultureInfo.InvariantCulture)}, \"n\": {n}}}");
                    return 0;
                }
                Console.Error.WriteLine("usage: VoxLat.Export.Probe sd|fields|bench <config.json> ...");
                return 2;
            }
            catch (Exception e)
            {
                Console.Error.WriteLine(e.ToString());
                return 1;
            }
        }
    }
}
