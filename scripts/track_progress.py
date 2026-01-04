"""
Track Daily Progress
"""
import json
from datetime import datetime
from pathlib import Path

def update_progress(task, status, notes=""):
    """Update daily progress"""
    
    progress_file = Path("progress_log.json")
    
    if progress_file.exists():
        with open(progress_file, "r") as f:
            progress = json.load(f)
    else:
        progress = {"days": []}
    
    day_entry = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "task": task,
        "status": status,  # "completed", "in_progress", "blocked"
        "notes": notes,
        "timestamp": datetime.now().isoformat()
    }
    
    progress["days"].append(day_entry)
    
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)
    
    print(f"✅ Progress updated: {task} - {status}")
    if notes:
        print(f"   Notes: {notes}")

def get_progress_summary():
    """Get progress summary"""
    
    progress_file = Path("progress_log.json")
    if not progress_file.exists():
        print("No progress log found.")
        return
    
    with open(progress_file, "r") as f:
        progress = json.load(f)
    
    days = progress.get("days", [])
    completed = len([d for d in days if d["status"] == "completed"])
    in_progress = len([d for d in days if d["status"] == "in_progress"])
    blocked = len([d for d in days if d["status"] == "blocked"])
    
    print("\n📊 Progress Summary:")
    print(f"   Total tasks: {len(days)}")
    print(f"   ✅ Completed: {completed}")
    print(f"   🔄 In Progress: {in_progress}")
    print(f"   ⚠️  Blocked: {blocked}")
    
    if days:
        print(f"\n📅 Recent tasks:")
        for day in days[-5:]:
            status_icon = "✅" if day["status"] == "completed" else "🔄" if day["status"] == "in_progress" else "⚠️"
            print(f"   {status_icon} {day['date']}: {day['task']}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        task = sys.argv[1]
        status = sys.argv[2] if len(sys.argv) > 2 else "completed"
        notes = sys.argv[3] if len(sys.argv) > 3 else ""
        update_progress(task, status, notes)
    else:
        get_progress_summary()

