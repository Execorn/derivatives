"""
Tests for cloud deployment configurations and Kubernetes manifests.
This validates the YAML syntax, schemas, and resource reservations.
"""
from __future__ import annotations

import os
from pathlib import Path
import yaml
import pytest

DEPLOY_DIR = Path(__file__).parents[1] / "deploy"


def test_docker_compose_yaml_syntax():
    """Verify that docker-compose.yml is valid YAML and has required services."""
    compose_path = DEPLOY_DIR / "docker-compose.yml"
    assert compose_path.exists(), "docker-compose.yml does not exist"

    with open(compose_path, "r") as f:
        data = yaml.safe_load(f)

    assert "services" in data, "services key is missing in docker-compose.yml"
    services = data["services"]

    # Verify key services are present
    required_services = ["kafka", "redis", "postgres", "api", "worker"]
    for service in required_services:
        assert service in services, f"Service {service} is missing in docker-compose.yml"

    # Verify worker has GPU reservation
    worker = services["worker"]
    assert "deploy" in worker, "deploy block missing in worker service"
    assert "resources" in worker["deploy"], "resources block missing in worker deploy"
    assert "reservations" in worker["deploy"]["resources"], "reservations block missing in worker resources"
    devices = worker["deploy"]["resources"]["reservations"].get("devices", [])
    assert len(devices) > 0, "No device reservation found for worker"
    assert devices[0].get("driver") == "nvidia", "NVIDIA driver not specified for worker GPU reservation"
    assert devices[0].get("capabilities") == ["gpu"], "GPU capability not specified for worker"

    # Verify Postgres connection pool variables are configured
    for s_name in ["api", "worker"]:
        env = services[s_name].get("environment", [])
        # Environment can be a list of strings or a dict
        env_dict = {}
        if isinstance(env, list):
            for item in env:
                if "=" in item:
                    k, v = item.split("=", 1)
                    env_dict[k] = v
        elif isinstance(env, dict):
            env_dict = env
        assert "PG_POOL_MAX_SIZE" in env_dict, f"PG_POOL_MAX_SIZE missing in {s_name} environment"


def test_k8s_manifests_exist():
    """Verify that all expected Kubernetes manifests exist in deploy/k8s/."""
    k8s_dir = DEPLOY_DIR / "k8s"
    assert k8s_dir.exists(), "deploy/k8s directory does not exist"

    expected_files = [
        "deployment.yaml",
        "scaledobject.yaml",
        "redis-deployment.yaml",
    ]
    for filename in expected_files:
        assert (k8s_dir / filename).exists(), f"{filename} is missing from deploy/k8s"


def test_k8s_deployment_syntax_and_constraints():
    """Verify deployment.yaml syntax, GPU constraints, nodeSelectors, tolerations and environment overrides."""
    deployment_path = DEPLOY_DIR / "k8s" / "deployment.yaml"
    with open(deployment_path, "r") as f:
        docs = list(yaml.safe_load_all(f))

    # Expect multiple documents: Namespace, Deployment (api), Service (api), Deployment (worker)
    assert len(docs) >= 4

    kinds = [doc["kind"] for doc in docs if doc is not None]
    assert "Namespace" in kinds
    assert "Deployment" in kinds
    assert "Service" in kinds

    worker_deployment = None
    api_deployment = None
    for doc in docs:
        if doc is not None and doc.get("kind") == "Deployment":
            if doc["metadata"]["name"] == "deepvol-worker":
                worker_deployment = doc
            elif doc["metadata"]["name"] == "deepvol-api":
                api_deployment = doc

    assert worker_deployment is not None, "deepvol-worker Deployment not found in deployment.yaml"
    assert api_deployment is not None, "deepvol-api Deployment not found in deployment.yaml"

    # Verify deepvol-worker pod specs
    pod_spec = worker_deployment["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]

    # Verify GPU resource requests & limits match exactly (1 GPU)
    resources = container.get("resources", {})
    limits = resources.get("limits", {})
    requests = resources.get("requests", {})
    assert limits.get("nvidia.com/gpu") == "1", "nvidia.com/gpu limit should be 1"
    assert requests.get("nvidia.com/gpu") == "1", "nvidia.com/gpu request should be 1"

    # Verify nodeSelector
    node_selector = pod_spec.get("nodeSelector", {})
    assert node_selector.get("cloud.google.com/gke-gpu") == "true", "Missing GKE GPU nodeSelector"
    assert node_selector.get("cloud.google.com/gke-accelerator") == "nvidia-l4", "Missing L4 accelerator nodeSelector"

    # Verify tolerations
    tolerations = pod_spec.get("tolerations", [])
    has_gpu_toleration = any(
        t.get("key") == "nvidia.com/gpu" and t.get("operator") == "Exists" and t.get("effect") == "NoSchedule"
        for t in tolerations
    )
    assert has_gpu_toleration, "Missing GPU NoSchedule toleration on worker deployment"

    # Verify SOTA optimizations in worker env
    env = container.get("env", [])
    env_vars = {item["name"]: item["value"] for item in env if "value" in item}
    assert env_vars.get("KAFKA_MAX_POLL_RECORDS") == "500"
    assert env_vars.get("KAFKA_FETCH_MIN_BYTES") == "1024"
    assert env_vars.get("KAFKA_FETCH_MAX_WAIT_MS") == "100"
    assert env_vars.get("KAFKA_ENABLE_AUTO_COMMIT") == "false"
    assert env_vars.get("PG_POOL_MAX_SIZE") == "20"

    # Verify API deployment does NOT request GPU
    api_container = api_deployment["spec"]["template"]["spec"]["containers"][0]
    api_resources = api_container.get("resources", {})
    api_limits = api_resources.get("limits", {})
    assert "nvidia.com/gpu" not in api_limits, "API gateway should not request GPU"
    
    api_pod_spec = api_deployment["spec"]["template"]["spec"]
    assert "nodeSelector" not in api_pod_spec, "API gateway should not have GPU nodeSelector"
    assert "tolerations" not in api_pod_spec, "API gateway should not have GPU tolerations"


def test_k8s_scaledobject_syntax():
    """Verify scaledobject.yaml syntax and targeting."""
    scaledobject_path = DEPLOY_DIR / "k8s" / "scaledobject.yaml"
    with open(scaledobject_path, "r") as f:
        so = yaml.safe_load(f)

    assert so["kind"] == "ScaledObject"
    assert so["metadata"]["name"] == "deepvol-worker-scaler"
    assert so["spec"]["scaleTargetRef"]["name"] == "deepvol-worker"
    
    # Check triggers
    triggers = so["spec"]["triggers"]
    assert len(triggers) > 0
    trigger = triggers[0]
    assert trigger["type"] == "kafka"
    metadata = trigger["metadata"]
    assert metadata["topic"] == "option-quotes"
    assert metadata["consumerGroup"] == "deepvol-calibrator-group"
    assert metadata["lagThreshold"] == "100"
    assert metadata["activationLagThreshold"] == "10"


def test_k8s_redis_deployment_syntax():
    """Verify redis-deployment.yaml syntax, resources, maxmemory configurations."""
    redis_path = DEPLOY_DIR / "k8s" / "redis-deployment.yaml"
    with open(redis_path, "r") as f:
        docs = list(yaml.safe_load_all(f))

    assert len(docs) >= 3
    kinds = [doc["kind"] for doc in docs if doc is not None]
    assert "PersistentVolumeClaim" in kinds
    assert "Deployment" in kinds
    assert "Service" in kinds

    redis_deployment = next(doc for doc in docs if doc is not None and doc.get("kind") == "Deployment")
    assert redis_deployment["metadata"]["name"] == "redis-master"
    
    # Verify resource limits
    redis_container = redis_deployment["spec"]["template"]["spec"]["containers"][0]
    assert redis_container["resources"]["limits"]["memory"] == "2Gi"
    
    # Verify command contains cache parameters
    cmd = redis_container["command"]
    assert "--maxmemory" in cmd
    assert "1536mb" in cmd
    assert "--maxmemory-policy" in cmd
    assert "allkeys-lru" in cmd
