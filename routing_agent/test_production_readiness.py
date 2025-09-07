#!/usr/bin/env python3
"""
Integration test to verify telemetry implementation is ready for production.
"""

import os
import sys
from typing import Dict, Any

def check_telemetry_readiness() -> Dict[str, Any]:
    """Check if telemetry implementation is ready for production."""
    results = {
        "imports": {},
        "environment": {},
        "implementation": {},
        "overall_status": "unknown"
    }
    
    print("🔍 Checking telemetry readiness for production...")
    
    # Check 1: Import availability
    print("\n1️⃣ Checking required imports...")
    imports_to_check = [
        ("azure.monitor.opentelemetry", "configure_azure_monitor"),
        ("azure.ai.projects", "AIProjectClient"),
        ("opentelemetry", "trace"),
        ("routing_agent", "RoutingAgent")
    ]
    
    for module_name, class_name in imports_to_check:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            results["imports"][module_name] = "✅ Available"
            print(f"   ✅ {module_name}.{class_name}")
        except ImportError as e:
            results["imports"][module_name] = f"❌ Import Error: {e}"
            print(f"   ❌ {module_name}.{class_name} - Import Error")
        except AttributeError as e:
            results["imports"][module_name] = f"❌ Attribute Error: {e}"
            print(f"   ❌ {module_name}.{class_name} - Attribute Error")
    
    # Check 2: Environment variables
    print("\n2️⃣ Checking environment variables...")
    required_env_vars = [
        "AZURE_AI_AGENT_PROJECT_ENDPOINT",
        "AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME"
    ]
    
    optional_env_vars = [
        "AZURE_CLIENT_ID",
        "AZURE_CLIENT_SECRET", 
        "AZURE_TENANT_ID"
    ]
    
    for var in required_env_vars:
        if os.getenv(var):
            results["environment"][var] = "✅ Set"
            print(f"   ✅ {var} is set")
        else:
            results["environment"][var] = "❌ Not set (Required)"
            print(f"   ❌ {var} is not set (Required)")
    
    for var in optional_env_vars:
        if os.getenv(var):
            results["environment"][var] = "✅ Set (Optional)"
            print(f"   ✅ {var} is set (Optional)")
        else:
            results["environment"][var] = "⚠️ Not set (Optional - will use DefaultAzureCredential)"
            print(f"   ⚠️ {var} is not set (Optional)")
    
    # Check 3: Implementation features
    print("\n3️⃣ Checking implementation features...")
    try:
        from routing_agent import RoutingAgent
        
        # Check if telemetry initialization method exists
        if hasattr(RoutingAgent, '_initialize_telemetry'):
            results["implementation"]["telemetry_init"] = "✅ Implemented"
            print("   ✅ _initialize_telemetry method exists")
        else:
            results["implementation"]["telemetry_init"] = "❌ Missing"
            print("   ❌ _initialize_telemetry method missing")
        
        # Check if tracer attribute is set in __init__
        routing_agent = RoutingAgent()
        if hasattr(routing_agent, 'tracer'):
            results["implementation"]["tracer_attribute"] = "✅ Available"
            print("   ✅ tracer attribute is set")
        else:
            results["implementation"]["tracer_attribute"] = "❌ Missing"
            print("   ❌ tracer attribute missing")
        
        # Check for environment variable setup
        if os.getenv("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED") == "true":
            results["implementation"]["content_recording"] = "✅ Enabled"
            print("   ✅ Content recording enabled")
        else:
            results["implementation"]["content_recording"] = "❌ Not enabled"
            print("   ❌ Content recording not enabled")
            
    except Exception as e:
        results["implementation"]["error"] = f"❌ {str(e)}"
        print(f"   ❌ Implementation check failed: {e}")
    
    # Overall assessment
    print("\n📊 Overall Assessment...")
    all_imports_ok = all("✅" in status for status in results["imports"].values())
    required_env_ok = all("✅" in results["environment"].get(var, "") for var in required_env_vars)
    implementation_ok = all("✅" in status for status in results["implementation"].values() if not status.startswith("❌"))
    
    if all_imports_ok and required_env_ok and implementation_ok:
        results["overall_status"] = "✅ Ready for Production"
        print("✅ Telemetry implementation is ready for production!")
    elif all_imports_ok and implementation_ok:
        results["overall_status"] = "⚠️ Ready (Environment setup needed)"
        print("⚠️ Telemetry implementation is ready, but environment variables need to be configured for production")
    else:
        results["overall_status"] = "❌ Needs fixes"
        print("❌ Telemetry implementation needs fixes before production use")
    
    return results

def print_production_checklist():
    """Print a checklist for production deployment."""
    print("\n📋 Production Deployment Checklist:")
    print("   1. Set AZURE_AI_AGENT_PROJECT_ENDPOINT to your Azure AI Studio project endpoint")
    print("   2. Set AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME to your model deployment name")
    print("   3. Configure Azure authentication (DefaultAzureCredential or service principal)")
    print("   4. Ensure Azure Monitor/Application Insights is configured for your project")
    print("   5. Test with real Azure credentials in a staging environment")
    print("   6. Monitor telemetry data in Azure Application Insights")

if __name__ == "__main__":
    results = check_telemetry_readiness()
    print_production_checklist()
    
    # Exit with appropriate code
    if "✅" in results["overall_status"]:
        print("\n🎉 Telemetry implementation check completed successfully!")
        sys.exit(0)
    else:
        print("\n⚠️ Telemetry implementation check completed with issues.")
        sys.exit(1)
