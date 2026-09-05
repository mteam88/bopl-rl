using System;
using System.IO;
using System.Reflection;
using BoplFixedMath;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;

public static class ReferenceKernel
{
    public static bool Enabled => !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("BOPL_KERNEL_INPUT"));
    private static bool completed;
    public static void Run()
    {
        if (completed) return;
        completed = true;
        string input = Environment.GetEnvironmentVariable("BOPL_KERNEL_INPUT");
        if (string.IsNullOrEmpty(input)) return;
        string output = Environment.GetEnvironmentVariable("BOPL_KERNEL_OUTPUT");
        try
        {
            using (var writer = new StreamWriter(output))
            {
                foreach (string line in File.ReadLines(input))
                {
                    JObject query = JObject.Parse(line);
                    JToken result;
                    try { result = Evaluate(query); }
                    catch (DivideByZeroException) { result = new JObject { ["error"] = "division_by_zero" }; }
                    catch (ArgumentOutOfRangeException) { result = new JObject { ["error"] = "domain" }; }
                    writer.WriteLine(result.ToString(Formatting.None));
                }
            }
            Debug.Log("REFERENCE_KERNEL_COMPLETE");
            Application.Quit(0);
        }
        catch (Exception error) { Debug.LogException(error); Application.Quit(2); }
    }

    private static Vec2 Vector(JToken value) => new Vec2(Fix.FromRaw((long)value[0]), Fix.FromRaw((long)value[1]));
    private static JArray Vector(Vec2 value) => new JArray(value.x.m_rawValue, value.y.m_rawValue);

    public static JToken Evaluate(JObject query)
    {
        if (((string)query["operation"]).StartsWith("dynamics_", StringComparison.Ordinal)) return ReferenceDynamics.Evaluate(query);
        var center = Vector(query["center"]);
        var extent = Vector(query["extent"]);
        var origin = Vector(query["origin"]);
        var basis = new Vec2(Fix.FromRaw((long)query["angle"]));
        var direction = new Vec2(Fix.FromRaw((long)query["direction"]));
        var radius = Fix.FromRaw((long)query["radius"]);
        var box = new Box { center = center,
            right = Vec2.ComplexMul(Vec2.right * extent.x, basis),
            up = Vec2.ComplexMul(Vec2.up * extent.y, basis),
            inverseExtents = new Vec2(Fix.One / extent.x, Fix.One / extent.y) };
        var circle = new Circle { center = center, radius = radius };
        var rounded = new RoundedRect { box = box, radius = radius };
        Fix near = Fix.Zero, far = Fix.Zero;
        bool hit;
        switch ((string)query["operation"])
        {
            case "surface_normal": case "surface_closest": case "surface_local": case "surface_position": case "surface_move":
                return SurfaceQuery(query, center, extent, origin, radius);
            case "circle_ray": hit = Raycast.RayCastCircle(origin, direction, circle, ref near, ref far); break;
            case "box_ray": hit = Raycast.RayCastBox(origin, direction, box, ref near, ref far); break;
            case "rounded_ray": hit = Raycast.RayCastRoundedRect(origin, direction, rounded, out near, out far); break;
            case "circle_overlap": return new JValue(circle.OverlapPoint(origin));
            case "box_overlap": return new JValue(box.OverlapPoint(origin));
            case "rounded_overlap": return new JValue(rounded.OverlapPoint(origin));
            case "closest": return Vector(rounded.GetClosestPoint(origin));
            default: throw new ArgumentException("unsupported kernel operation");
        }
        return hit ? new JObject { ["hit"] = true, ["near"] = near.m_rawValue, ["far"] = far.m_rawValue }
            : new JObject { ["hit"] = false };
    }
    private static JToken SurfaceQuery(JObject query, Vec2 center, Vec2 extent, Vec2 origin, Fix radius)
    {
        var surface = UnityEngine.Object.FindObjectOfType<StickyRoundedRectangle>();
        var body = surface.GetGroundBody();
        body.position = center;
        body.rotation = Fix.FromRaw((long)query["angle"]);
        surface.rr.radius = radius;
        var flags = BindingFlags.Instance | BindingFlags.NonPublic;
        typeof(StickyRoundedRectangle).GetField("extents", flags).SetValue(surface, extent);
        var segments = new Fix[] {extent.x*(Fix)2L,radius*Fix.PiOver2,extent.y*(Fix)2L,radius*Fix.PiOver2,
            extent.x*(Fix)2L,radius*Fix.PiOver2,extent.y*(Fix)2L,radius*Fix.PiOver2};
        typeof(StickyRoundedRectangle).GetField("circumferences", flags).SetValue(surface, segments);
        typeof(StickyRoundedRectangle).GetField("totalCircum", flags).SetValue(surface, segments[0]*(Fix)2L+segments[1]*(Fix)4L+segments[2]*(Fix)2L);
        Fix playerRadius = Fix.FromRaw((long)query["player_radius"]);
        Fix local = Fix.FromRaw((long)query["local"]);
        switch ((string)query["operation"])
        {
            case "surface_normal": return Vector(surface.currentNormal(origin));
            case "surface_closest": return Vector(surface.GetClosestPoint(origin));
            case "surface_local": return new JValue(surface.LocalPlayerPosFromPos(origin, playerRadius).m_rawValue);
            case "surface_position": return Vector(surface.PositionFromLocalPlayerPos(local, playerRadius));
            case "surface_move": return new JValue(surface.moveAlongSurface(null, playerRadius, Fix.FromRaw((long)query["speed"]), GameTime.FixedTimeStep, local).m_rawValue);
            default: throw new ArgumentException("surface operation");
        }
    }

}
