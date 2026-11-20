# Render API Load-Test Comparison

## Objective

Measure how the deployed FastAPI service responds as the number of concurrent Locust users increases. This experiment changes the **client-side concurrency**, not the number of server containers.

## Test environment and method

- Target: deployed Render API
- Load tool: Locust
- Concurrent users: 1, 2, and 5
- Duration: approximately 2 minutes per run
- Tested operations: health checks, single-audio prediction, and batch prediction
- Result source: Locust CSV exports stored in this directory

## Aggregate results

| Concurrent users | Requests | Failures | Average response time | Median response time | 95th percentile | Maximum response time | Throughput |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 39 | 0 | 893 ms | 700 ms | 1,900 ms | 2,003 ms | 0.335 req/s |
| 2 | 81 | 0 | 948 ms | 670 ms | 2,300 ms | 3,443 ms | 0.678 req/s |
| 5 | 205 | 0 | 836 ms | 730 ms | 1,900 ms | 2,734 ms | 1.708 req/s |

## Findings

- No request failures were recorded in any of the three runs.
- Throughput increased from 0.335 requests/second with one user to 1.708 requests/second with five users, approximately a 5.1-fold increase.
- Average response time did not increase monotonically as concurrency increased. The five-user run had the lowest measured average, while its median was slightly higher than the other runs.
- Batch prediction was the slowest operation in the test mix, as it processes multiple audio files per request.

## Interpretation and limitations

The results show that the deployed API handled the tested workloads without recorded failures and processed more requests per second as Locust concurrency increased. They do **not** establish the maximum production capacity of the service.

Each run lasted only about two minutes and used a small number of virtual users. The request mix, Render instance state, cold starts, network conditions, and run-to-run variation can affect the measurements. These results should therefore be treated as a small performance experiment, not a capacity guarantee.

This Render experiment is separate from the Docker container comparison. Render tests varied the number of concurrent Locust users against the hosted service; the Docker experiment varied the number of API containers behind an Nginx load balancer.

## Reproducibility

The Locust test script is in [../locustfile.py](../locustfile.py). The exported CSV files in this directory provide the underlying measurements. Keep the original exports unchanged when possible.
