using System;

namespace UnityEngine
{
	// Data-only stub: gameplay toggles emission/plays effects; nothing reads back.
	public class ParticleSystem : Component
	{
		private MainModule mainModule;
		private EmissionModule emissionModule;
		private ShapeModule shapeModule;
		private VelocityOverLifetimeModule velocityModule;
		private ColorOverLifetimeModule colorModule;
		private SizeOverLifetimeModule sizeModule;
		private NoiseModule noiseModule;
		private TrailModule trailsModule;
		private TextureSheetAnimationModule textureSheetAnimationModule;

		public bool isPlaying { get; private set; }
		public bool isStopped => !isPlaying;
		public bool isEmitting => isPlaying;
		public bool isPaused { get; private set; }
		public int particleCount => 0;
		public float time { get; set; }
		public uint randomSeed { get; set; }
		public bool useAutoRandomSeed { get; set; } = true;

		public ParticleSystem()
		{
			mainModule = new MainModule(this);
			emissionModule = new EmissionModule(this);
			shapeModule = new ShapeModule(this);
			velocityModule = new VelocityOverLifetimeModule(this);
			colorModule = new ColorOverLifetimeModule(this);
			sizeModule = new SizeOverLifetimeModule(this);
			noiseModule = new NoiseModule(this);
			trailsModule = new TrailModule(this);
			textureSheetAnimationModule = new TextureSheetAnimationModule(this);
		}

		public MainModule main => mainModule;
		public EmissionModule emission => emissionModule;
		public ShapeModule shape => shapeModule;
		public VelocityOverLifetimeModule velocityOverLifetime => velocityModule;
		public ColorOverLifetimeModule colorOverLifetime => colorModule;
		public SizeOverLifetimeModule sizeOverLifetime => sizeModule;
		public NoiseModule noise => noiseModule;
		public TrailModule trails => trailsModule;
		public TextureSheetAnimationModule textureSheetAnimation => textureSheetAnimationModule;

		public void Play() => isPlaying = true;
		public void Play(bool withChildren) => isPlaying = true;
		public void Stop() => isPlaying = false;
		public void Stop(bool withChildren) => isPlaying = false;
		public void Stop(bool withChildren, ParticleSystemStopBehavior stopBehavior) => isPlaying = false;
		public void Pause() => isPaused = true;
		public void Clear() { }
		public void Clear(bool withChildren) { }
		public void Emit(int count) { }
		public void Emit(EmitParams emitParams, int count) { }
		public bool IsAlive() => isPlaying;
		public bool IsAlive(bool withChildren) => isPlaying;
		public void Simulate(float t, bool withChildren = true, bool restart = true, bool fixedTimeStep = true) { }

		// Legacy accessors
		public Color startColor { get; set; } = Color.white;
		public float startSize { get; set; } = 1f;
		public float startSpeed { get; set; } = 5f;
		public float startLifetime { get; set; } = 5f;
		public bool loop { get; set; }
		public bool playOnAwake { get; set; } = true;
		public float emissionRate { get; set; }
		public int GetParticles(Particle[] particles) => 0;
		public void SetParticles(Particle[] particles, int size) { }

		// ---- modules -------------------------------------------------------

		public class MainModule
		{
			internal MainModule(ParticleSystem ps) { }
			public float duration { get; set; } = 5f;
			public bool loop { get; set; }
			public MinMaxCurve startLifetime { get; set; } = 5f;
			public MinMaxCurve startSpeed { get; set; } = 5f;
			public MinMaxCurve startSize { get; set; } = 1f;
			public MinMaxCurve startSizeX { get; set; } = 1f;
			public MinMaxCurve startSizeY { get; set; } = 1f;
			public MinMaxCurve startRotation { get; set; }
			public MinMaxGradient startColor { get; set; } = Color.white;
			public MinMaxCurve gravityModifier { get; set; }
			public ParticleSystemSimulationSpace simulationSpace { get; set; }
			public float simulationSpeed { get; set; } = 1f;
			public int maxParticles { get; set; } = 1000;
			public bool playOnAwake { get; set; } = true;
			public bool prewarm { get; set; }
			public ParticleSystemScalingMode scalingMode { get; set; }
			public ParticleSystemStopAction stopAction { get; set; }
			public bool useUnscaledTime { get; set; }
		}

		public class EmissionModule
		{
			internal EmissionModule(ParticleSystem ps) { }
			public bool enabled { get; set; } = true;
			public MinMaxCurve rateOverTime { get; set; } = 10f;
			public MinMaxCurve rateOverDistance { get; set; }
			public int burstCount { get; set; }
			public void SetBursts(Burst[] bursts) { }
			public void SetBurst(int index, Burst burst) { }
			public Burst GetBurst(int index) => default;
		}

		public class ShapeModule
		{
			internal ShapeModule(ParticleSystem ps) { }
			public bool enabled { get; set; } = true;
			public ParticleSystemShapeType shapeType { get; set; }
			public float radius { get; set; } = 1f;
			public float angle { get; set; } = 25f;
			public float arc { get; set; } = 360f;
			public Vector3 scale { get; set; } = Vector3.one;
			public Vector3 position { get; set; }
			public Vector3 rotation { get; set; }
			public SpriteRenderer spriteRenderer { get; set; }
			public Sprite sprite { get; set; }
			public MeshRenderer meshRenderer { get; set; }
			public Mesh mesh { get; set; }
		}

		public class VelocityOverLifetimeModule
		{
			internal VelocityOverLifetimeModule(ParticleSystem ps) { }
			public bool enabled { get; set; }
			public MinMaxCurve x { get; set; }
			public MinMaxCurve y { get; set; }
			public MinMaxCurve z { get; set; }
			public MinMaxCurve orbitalX { get; set; }
			public MinMaxCurve orbitalY { get; set; }
			public MinMaxCurve orbitalZ { get; set; }
			public MinMaxCurve orbitalOffsetX { get; set; }
			public MinMaxCurve orbitalOffsetY { get; set; }
			public MinMaxCurve orbitalOffsetZ { get; set; }
			public MinMaxCurve radial { get; set; }
			public MinMaxCurve speedModifier { get; set; } = 1f;
			public ParticleSystemSimulationSpace space { get; set; }
		}

		public class ColorOverLifetimeModule
		{
			internal ColorOverLifetimeModule(ParticleSystem ps) { }
			public bool enabled { get; set; }
			public MinMaxGradient color { get; set; }
		}

		public class SizeOverLifetimeModule
		{
			internal SizeOverLifetimeModule(ParticleSystem ps) { }
			public bool enabled { get; set; }
			public MinMaxCurve size { get; set; } = 1f;
		}

		public class NoiseModule
		{
			internal NoiseModule(ParticleSystem ps) { }
			public bool enabled { get; set; }
			public MinMaxCurve strength { get; set; } = 1f;
			public float frequency { get; set; } = 0.5f;
		}

		public class TrailModule
		{
			internal TrailModule(ParticleSystem ps) { }
			public bool enabled { get; set; }
			public MinMaxCurve lifetime { get; set; } = 1f;
			public MinMaxCurve widthOverTrail { get; set; } = 1f;
		}

		public class TextureSheetAnimationModule
		{
			internal TextureSheetAnimationModule(ParticleSystem ps) { }
			public bool enabled { get; set; }
			public int numTilesX { get; set; } = 1;
			public int numTilesY { get; set; } = 1;
			public MinMaxCurve frameOverTime { get; set; }
			public MinMaxCurve startFrame { get; set; }
		}

		[Serializable]
		public struct MinMaxCurve
		{
			public ParticleSystemCurveMode mode;
			public float constant;
			public float constantMin, constantMax;
			public AnimationCurve curve, curveMin, curveMax;
			public float curveMultiplier;

			public MinMaxCurve(float constant) : this()
			{
				mode = ParticleSystemCurveMode.Constant;
				this.constant = constant;
				constantMin = constantMax = constant;
			}

			public MinMaxCurve(float min, float max) : this()
			{
				mode = ParticleSystemCurveMode.TwoConstants;
				constantMin = min; constantMax = max; constant = max;
			}

			public MinMaxCurve(float multiplier, AnimationCurve curve) : this()
			{
				mode = ParticleSystemCurveMode.Curve;
				curveMultiplier = multiplier; this.curve = curve;
			}

			public float Evaluate(float time) => constant;
			public static implicit operator MinMaxCurve(float constant) => new MinMaxCurve(constant);
		}

		[Serializable]
		public struct MinMaxGradient
		{
			public ParticleSystemGradientMode mode;
			public Color color;
			public Color colorMin, colorMax;
			public Gradient gradient, gradientMin, gradientMax;

			public MinMaxGradient(Color color) : this()
			{
				mode = ParticleSystemGradientMode.Color;
				this.color = color; colorMin = colorMax = color;
			}

			public MinMaxGradient(Gradient gradient) : this()
			{
				mode = ParticleSystemGradientMode.Gradient;
				this.gradient = gradient;
			}

			public MinMaxGradient(Color min, Color max) : this()
			{
				mode = ParticleSystemGradientMode.TwoColors;
				colorMin = min; colorMax = max; color = max;
			}

			public Color Evaluate(float time) => color;
			public static implicit operator MinMaxGradient(Color color) => new MinMaxGradient(color);
			public static implicit operator MinMaxGradient(Gradient gradient) => new MinMaxGradient(gradient);
		}

		[Serializable]
		public struct Burst
		{
			public float time;
			public MinMaxCurve count;
			public int cycleCount;
			public float repeatInterval;
			public Burst(float time, short count) : this() { this.time = time; this.count = count; cycleCount = 1; repeatInterval = 0.01f; }
			public Burst(float time, MinMaxCurve count) : this() { this.time = time; this.count = count; cycleCount = 1; repeatInterval = 0.01f; }
		}

		public struct Particle
		{
			public Vector3 position { get; set; }
			public Vector3 velocity { get; set; }
			public float remainingLifetime { get; set; }
			public float startLifetime { get; set; }
			public float startSize { get; set; }
			public Color32 startColor { get; set; }
			public float rotation { get; set; }
		}

		public struct EmitParams
		{
			public Vector3 position { get; set; }
			public Vector3 velocity { get; set; }
			public float startLifetime { get; set; }
			public float startSize { get; set; }
			public Color32 startColor { get; set; }
			public bool applyShapeToPosition { get; set; }
		}
	}

	public enum ParticleSystemSimulationSpace { Local, World, Custom }
	public enum ParticleSystemScalingMode { Hierarchy, Local, Shape }
	public enum ParticleSystemStopBehavior { StopEmittingAndClear, StopEmitting }
	public enum ParticleSystemStopAction { None, Disable, Destroy, Callback }
	public enum ParticleSystemCurveMode { Constant, Curve, TwoCurves, TwoConstants }
	public enum ParticleSystemGradientMode { Color, Gradient, TwoColors, TwoGradients, RandomColor }
	public enum ParticleSystemShapeType { Sphere, Hemisphere = 2, Cone = 4, Box = 5, Mesh = 6, MeshRenderer = 13, SpriteRenderer = 15, Circle = 10, Edge = 12, Donut = 17, Rectangle = 18, Sprite = 14 }

	public class ParticleSystemRenderer : Renderer
	{
		public ParticleSystemRenderMode renderMode { get; set; }
		public float lengthScale { get; set; } = 2f;
		public float velocityScale { get; set; }
	}

	public enum ParticleSystemRenderMode { Billboard, Stretch, HorizontalBillboard, VerticalBillboard, Mesh, None }
}
