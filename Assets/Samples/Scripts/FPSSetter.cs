using UnityEngine;

public class FpsSetter : MonoBehaviour
{
    [SerializeField] private int _targetFps = 10000;

    private void Awake()
    {
        SetFps();
    }

    private void SetFps()
    {
        QualitySettings.vSyncCount  = 0;
        Application.targetFrameRate = _targetFps;
    }
}