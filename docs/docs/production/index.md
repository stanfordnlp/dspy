# Deploying DSPy

The tooling we ship for running DSPy programs in production: observability, reproducibility, deployment, and scalability.

!!! info "Looking for who's using DSPy?"

    DSPy is in production at **Shopify, Databricks, Dropbox, JetBlue, Moody's, Replit, AWS, Sephora, VMware**, and dozens more. [:octicons-arrow-right-24: See companies using DSPy](../community/use-cases.md)

<div class="grid cards" style="text-align: left;" markdown>

- :material-magnify-expand:{ .lg .middle } __Monitoring & Observability__

    ---

    Monitor your DSPy programs using **MLflow Tracing**, based on OpenTelemetry.

    [:octicons-arrow-right-24: Set Up Observability](../tutorials/observability/index.md#tracing)

- :material-ab-testing: __Reproducibility__

    ---

    Log programs, metrics, configs, and environments for full reproducibility with DSPy's native MLflow integration.

    [:octicons-arrow-right-24: MLflow Integration](https://mlflow.org/docs/latest/llms/dspy/index.html)

- :material-rocket-launch: __Deployment with MLflow__

    ---

    When it's time to productionize, deploy your application easily with DSPy's integration with MLflow Model Serving.

    [:octicons-arrow-right-24: Deployment with MLflow](../tutorials/deployment/index.md)

- :material-arrow-up-right-bold: __Scalability__

    ---

    DSPy is designed with thread-safety in mind and offers native asynchronous execution support for high-throughput environments.

    [:octicons-arrow-right-24: Async Program](../api/utils/asyncify.md)

</div>
