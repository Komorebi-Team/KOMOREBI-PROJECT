from src.modeling.lightgbm_workflow import run_lightgbm_feature_ablation_study


def main():
    results = run_lightgbm_feature_ablation_study()
    print("Comparativa de variantes LightGBM terminada.")
    print(results["comparison"].to_string(index=False))
    print(f"\nResumen guardado en: {results['comparison_path']}")


if __name__ == "__main__":
    main()
