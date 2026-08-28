using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

// Smash-style stock counter: one row of mini slime silhouettes per player,
// anchored to a screen corner (P1 top-left, P2 top-right, P3 bottom-left,
// P4 bottom-right). Purely visual; reads Player.StocksRemaining every frame.
public class StockPips : MonoBehaviour
{
	private const float PipSize = 52f;

	private const float PipSpacing = 58f;

	private const float EdgeMargin = 46f;

	private Player player;

	private int cornerIndex;

	private readonly List<Image> pips = new List<Image>();

	private SlimeController slimeRef;

	private Sprite pipSprite;

	private Material pipMaterial;

	private static Canvas hudCanvas;

	public static void CreateForPlayers(List<Player> players, SlimeController[] slimes)
	{
		if (Settings.Get() == null || Settings.Get().Stocks <= 0)
		{
			return;
		}
		GameObject gameObject = new GameObject("StockHUD");
		hudCanvas = gameObject.AddComponent<Canvas>();
		hudCanvas.renderMode = RenderMode.ScreenSpaceOverlay;
		hudCanvas.sortingOrder = 40;
		CanvasScaler canvasScaler = gameObject.AddComponent<CanvasScaler>();
		canvasScaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
		canvasScaler.referenceResolution = new Vector2(1920f, 1080f);
		for (int i = 0; i < players.Count; i++)
		{
			GameObject obj = new GameObject("StockPips_P" + (players[i].Id + 1));
			RectTransform rectTransform = obj.AddComponent<RectTransform>();
			rectTransform.SetParent(gameObject.transform, worldPositionStays: false);
			rectTransform.anchorMin = Vector2.zero;
			rectTransform.anchorMax = Vector2.one;
			rectTransform.offsetMin = Vector2.zero;
			rectTransform.offsetMax = Vector2.zero;
			StockPips stockPips = obj.AddComponent<StockPips>();
			stockPips.player = players[i];
			stockPips.cornerIndex = i;
			stockPips.slimeRef = ((slimes != null && i < slimes.Length) ? slimes[i] : null);
			// Render pips with the player's own slime material so they match
			// the player's colors exactly.
			stockPips.pipMaterial = players[i].Color;
		}
	}

	private void LateUpdate()
	{
		// After a respawn the old SlimeController is destroyed; re-find the
		// player's new body so the pips keep mirroring it.
		if ((slimeRef == null || slimeRef.IsDestroyed) && player != null && player.IsAlive)
		{
			SlimeController[] array = Object.FindObjectsOfType<SlimeController>();
			for (int k = 0; k < array.Length; k++)
			{
				if (array[k].playerNumber == player.Id && !array[k].IsDestroyed)
				{
					slimeRef = array[k];
					break;
				}
			}
		}
		// Live-copy the slime's current sprite so the pips are mini copies of
		// the player.
		if (slimeRef != null && !slimeRef.IsDestroyed && slimeRef.GetPlayerSprite() != null && slimeRef.GetPlayerSprite().sprite != null)
		{
			pipSprite = slimeRef.GetPlayerSprite().sprite;
			for (int j = 0; j < pips.Count; j++)
			{
				pips[j].sprite = pipSprite;
			}
		}
		if (pipSprite == null)
		{
			return;
		}
		int num = ((player != null && player.IsAlive) ? (player.StocksRemaining + 1) : 0);
		while (pips.Count < num)
		{
			pips.Add(CreatePip(pips.Count));
		}
		for (int i = 0; i < pips.Count; i++)
		{
			pips[i].enabled = i < num;
		}
	}

	private Image CreatePip(int index)
	{
		GameObject obj = new GameObject("pip" + index);
		obj.transform.SetParent(base.transform, worldPositionStays: false);
		Image image = obj.AddComponent<Image>();
		image.sprite = pipSprite;
		if (pipMaterial != null)
		{
			image.material = pipMaterial;
		}
		image.preserveAspect = true;
		image.raycastTarget = false;
		RectTransform rectTransform = image.rectTransform;
		bool flag = cornerIndex % 2 == 0;
		bool flag2 = cornerIndex < 2;
		Vector2 vector = new Vector2(flag ? 0f : 1f, flag2 ? 1f : 0f);
		rectTransform.anchorMin = vector;
		rectTransform.anchorMax = vector;
		rectTransform.pivot = vector;
		float num = EdgeMargin + (float)index * PipSpacing;
		rectTransform.anchoredPosition = new Vector2(flag ? num : (0f - num), flag2 ? (0f - EdgeMargin) : EdgeMargin);
		rectTransform.sizeDelta = new Vector2(PipSize, PipSize);
		return image;
	}
}
