import subprocess
import os
import time

scripts = [
    # ۱. مرحله تحقیق و بصری‌سازی (Research & Visuals)
    "experiments/visualizations/cluster_optimization_audit.py",
    "experiments/visualizations/cluster_formation_viz.py",
    "experiments/visualizations/smote_balancing_viz.py",
    
    # ۲. مرحله تحلیل پرسونا و ریسک (Prototypes)
    "experiments/prototypes/3_discovery_persona_analysis.py",
    "experiments/prototypes/business_impact_optimizer.py",
    
    # ۳. مرحله بنچ‌مارک و مقایسه فنی (Benchmarking)
    "experiments/benchmarking/model_benchmark.py",
    "experiments/benchmarking/sampling_strategy_assessment.py",
    "experiments/benchmarking/sensitivity_analysis_audit.py", 
    "experiments/benchmarking/run_full_analysis.py",
    
    # ۴. بصری‌سازی مدل و اعتبارسنجی نهایی
    "experiments/visualizations/logistic_boundary_viz.py",
    "experiments/prototypes/ensemble_risk_validator.py",
    
    # ۵. سیستم نهایی عملیاتی و تاییدیه ریاضی (Core & Docs)
    "core_system/production_engine.py",
    "docs/model_verification_audit.py"
]

def run_all_modules():
    print("="*60)
    print("🚀 STARTING FULL PROJECT EXECUTION PIPELINE")
    print("Course: Computer Applications in Business Systems")
    print("="*60)
    
    root_path = os.getcwd()
    start_time = time.time()
    success_count = 0

    for script in scripts:
        script_full_path = os.path.join(root_path, script)
        
        if os.path.exists(script_full_path):
            script_dir = os.path.dirname(script_full_path)
            script_name = os.path.basename(script_full_path)
            
            print(f"\n[RUNNING] {script_name}")
            print(f"Location: {script}")
            
            try:
                subprocess.run(["python", script_name], cwd=script_dir, check=True)
                print(f"✅ SUCCESS: {script_name} finished.")
                success_count += 1
            except subprocess.CalledProcessError:
                print(f"❌ ERROR: {script_name} failed during execution.")
            except Exception as e:
                print(f"⚠️ UNEXPECTED ERROR in {script_name}: {e}")
        else:
            print(f"\n[MISSING] File not found: {script}")

    end_time = time.time()
    duration = end_time - start_time

    print("\n" + "="*60)
    print(f"🏁 PIPELINE FINISHED")
    print(f"Total Time: {duration:.2f} seconds")
    print(f"Successful Modules: {success_count}/{len(scripts)}")
    print("Check the 'outputs/' folder for all plots, GIFs, and CSV reports.")
    print("="*60)

if __name__ == "__main__":
    run_all_modules()