#!/usr/bin/env python3
"""
Simple deployment script for MentorEval Leaderboard
Copies files to repository root for GitHub Pages deployment
"""

import os
import shutil
from pathlib import Path

def deploy():
    """Deploy the webpage to repository root."""
    print("🚀 Deploying MentorEval Leaderboard...")
    print(f"📁 Current directory: {os.getcwd()}")
    
    # Source and destination paths
    webpage_dir = Path("webpage")
    root_dir = Path(".")
    
    print(f"📁 Webpage directory: {webpage_dir.absolute()}")
    print(f"📁 Root directory: {root_dir.absolute()}")
    print(f"📁 Webpage exists: {webpage_dir.exists()}")
    if webpage_dir.exists():
        print(f"📁 Webpage contents: {list(webpage_dir.iterdir())}")
    
    # Files to copy
    files_to_copy = [
        "index.html",
        "style.css", 
        "script.js"
    ]
    
    # Copy files and fix paths for root directory serving
    for file_name in files_to_copy:
        src = webpage_dir / file_name
        dst = root_dir / file_name
        
        if src.exists():
            if file_name == "script.js":
                # Fix paths for root directory serving
                with open(src, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Convert ../ paths to direct paths for root serving
                content = content.replace('../results/', 'results/')
                content = content.replace('../runs/', 'runs/')
                with open(dst, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ Copied and fixed paths in {file_name}")
            else:
                shutil.copy2(src, dst)
                print(f"✅ Copied {file_name}")
        else:
            print(f"❌ {file_name} not found in webpage/")
    
    # Copy assets directory
    assets_src = webpage_dir / "assets"
    assets_dst = root_dir / "assets"
    
    if assets_src.exists():
        if assets_dst.exists():
            shutil.rmtree(assets_dst)
        shutil.copytree(assets_src, assets_dst)
        print("✅ Copied assets/")
    else:
        print("❌ assets/ directory not found")
    
    print("🎉 Deployment complete!")
    print("📁 Files are ready for GitHub Pages")
    print("💡 Commit and push to deploy")

if __name__ == "__main__":
    deploy()
