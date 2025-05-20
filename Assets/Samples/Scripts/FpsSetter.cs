using UnityEngine;

public class FpsSetter : MonoBehaviour
{
    [SerializeField] private int _targetFPS = 10000;

    private void Awake()
    {
        SetFPS();
    }

    private void SetFPS()
    {
        QualitySettings.vSyncCount  = 0;
        Application.targetFrameRate = _targetFPS;
    }
}