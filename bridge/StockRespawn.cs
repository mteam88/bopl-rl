using System.Collections.Generic;
using BoplFixedMath;
using UnityEngine;

// Smash-style stocks (Settings.Stocks lives per round). When a player's last
// body dies with a stock left, three respawn points appear at the top of the
// screen; the player aims left/center/right to pick one and confirms with
// jump, reviving instantly at that point (no travel animation, so the respawn
// can't be spawn-camped).
public static class StockRespawn
{
	private static GameObject cachedIndicatorPrefab;

	public static bool CanConsumeStock(Player player, SlimeController slime)
	{
		if (Settings.Get() == null || Settings.Get().Stocks <= 0)
		{
			return false;
		}
		if (player.StocksRemaining <= 0 || GameSessionHandler.SuddenDeathInProgress || GameSessionHandler.HasGameEnded())
		{
			return false;
		}
		return GetIndicatorPrefab(slime) != null;
	}

	// The revive position indicator prefab is referenced by the Revival
	// ability (QuickSpell.spell), reachable through the ability registry that
	// every SlimeController carries.
	public static GameObject GetIndicatorPrefab(SlimeController slime)
	{
		if (cachedIndicatorPrefab != null)
		{
			return cachedIndicatorPrefab;
		}
		NamedSpriteList namedSpriteList = ((slime != null) ? slime.abilityIcons : null);
		if (namedSpriteList == null)
		{
			return null;
		}
		for (int i = 0; i < namedSpriteList.sprites.Count; i++)
		{
			NamedSprite namedSprite = namedSpriteList.sprites[i];
			if (namedSprite.associatedGameObject == null)
			{
				continue;
			}
			QuickSpell component = namedSprite.associatedGameObject.GetComponent<QuickSpell>();
			if (component != null && component.GetComponent<Revive>() != null && component.SpellPrefab != null)
			{
				cachedIndicatorPrefab = component.SpellPrefab;
				return cachedIndicatorPrefab;
			}
		}
		return null;
	}

	// Safe parking spot for the hidden clone while the player picks a respawn
	// point: the top-center point, well above water (a clone left at the death
	// position could drown again and burn every remaining stock).
	public static Vec2 CloneHoldPosition()
	{
		return StockRespawnChooser.ComputeRespawnPoints()[1];
	}

	public static void BeginChoice(Player player, SlimeController deadSlime, SlimeController clone)
	{
		GameObject obj = new GameObject("StockRespawnChooser");
		StockRespawnChooser stockRespawnChooser = obj.AddComponent<StockRespawnChooser>();
		stockRespawnChooser.Setup(player, GetIndicatorPrefab(deadSlime), clone);
	}
}

public class StockRespawnChooser : MonoUpdatable
{
	// Generous fallback so the player can time their re-entry around enemies;
	// jump confirms whenever they're ready.
	private const int ChoiceTicks = 360;

	private Player player;

	private SlimeController clone;

	private RevivePositionIndicator[] indicators;

	private int selected = 1;

	private int ticksLeft;

	private bool finished;

	private void Awake()
	{
		Updater.RegisterUpdatable(this);
	}

	public override void Init()
	{
	}

	public void Setup(Player player, GameObject indicatorPrefab, SlimeController clone)
	{
		this.player = player;
		this.clone = clone;
		ticksLeft = ChoiceTicks;
		Vec2[] spawnPoints = ComputeRespawnPoints();
		indicators = new RevivePositionIndicator[spawnPoints.Length];
		for (int i = 0; i < spawnPoints.Length; i++)
		{
			GameObject gameObject = FixTransform.InstantiateFixed(indicatorPrefab, spawnPoints[i]);
			indicators[i] = gameObject.GetComponent<RevivePositionIndicator>();
			indicators[i].SetPlayerId(player.Id);
		}
	}

	// Three points spread across the top of the arena, derived from the
	// level's player spawns so they adapt per level (deterministic; no
	// dependence on the visual camera).
	internal static Vec2[] ComputeRespawnPoints()
	{
		Vec2[] array = GameSessionHandler.playerSpawns_readonly;
		Fix fix = (Fix)(-12L);
		Fix fix2 = (Fix)12L;
		Fix fix3 = (Fix)8L;
		if (array != null && array.Length != 0)
		{
			fix = array[0].x;
			fix2 = array[0].x;
			fix3 = array[0].y;
			for (int i = 1; i < array.Length; i++)
			{
				fix = Fix.Min(fix, array[i].x);
				fix2 = Fix.Max(fix2, array[i].x);
				fix3 = Fix.Max(fix3, array[i].y);
			}
		}
		Fix fix4 = (fix + fix2) / (Fix)2L;
		Fix fix5 = Fix.Max((fix2 - fix) / (Fix)2L, (Fix)14L);
		Fix y = Fix.Min(fix3 + (Fix)12L, SceneBounds.Camera_YMax - (Fix)6L);
		return new Vec2[3]
		{
			new Vec2(fix4 - fix5, y),
			new Vec2(fix4, y),
			new Vec2(fix4 + fix5, y)
		};
	}

	public override void UpdateSim(Fix simDeltaTime)
	{
		if (finished)
		{
			return;
		}
		if (clone == null || clone.IsDestroyed)
		{
			Abort();
			return;
		}
		Vec2 vec = player.InputVector();
		Fix fix = (Fix)0.3f;
		if (vec.x < -fix)
		{
			selected = 0;
		}
		else if (vec.x > fix)
		{
			selected = 2;
		}
		else if (Fix.Abs(vec.y) > fix)
		{
			selected = 1;
		}
		UpdateHighlight();
		ticksLeft--;
		if (ticksLeft <= 0 || player.jumpButton_PressedThisFrame())
		{
			Finish();
		}
	}

	// Visual-only feedback: the chosen point grows, the others shrink.
	private void UpdateHighlight()
	{
		for (int i = 0; i < indicators.Length; i++)
		{
			if (!(indicators[i] == null) && !indicators[i].IsDestroyed)
			{
				float target = ((i == selected) ? 1.45f : 0.75f);
				Transform obj = indicators[i].transform;
				float num = Mathf.MoveTowards(obj.localScale.x, target, Time.deltaTime * 6f + 0.06f);
				obj.localScale = new Vector3(num, num, 1f);
			}
		}
	}

	// Instant revive at the chosen point: no wisp travel animation, so
	// opponents get no telegraph to camp.
	private void Finish()
	{
		finished = true;
		RevivePositionIndicator revivePositionIndicator = indicators[selected];
		Vec2 pos = revivePositionIndicator.GetPos();
		for (int i = 0; i < indicators.Length; i++)
		{
			if (indicators[i] != null && !indicators[i].IsDestroyed)
			{
				indicators[i].End();
			}
		}
		AudioManager.Get().Play("reviveDing");
		clone.ReSpawnFast(pos);
		Updater.DestroyFix(base.gameObject);
	}

	private void Abort()
	{
		finished = true;
		for (int i = 0; i < indicators.Length; i++)
		{
			if (indicators[i] != null && !indicators[i].IsDestroyed)
			{
				indicators[i].End();
			}
		}
		Updater.DestroyFix(base.gameObject);
	}
}
