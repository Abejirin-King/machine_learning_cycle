# Docker Container Scaling Results

## Objective

The experiment compared the voice-command FastAPI service with one, two, and five API containers behind an Nginx reverse proxy. The purpose was to observe whether increasing the number of API replicas improved request throughput and response latency.

## Test environment and method

- Container environment: Killercoda Docker playground.
- API image: `voice-command-api:v1.2.0`.
- Reverse proxy: Nginx, with a configuration matching the number of API replicas.
- Load generator: Locust running from the Windows development machine.
- Concurrent users: 5.
- Spawn rate: 1 user per second.
- Test duration: 2 minutes per configuration.
- Tested endpoints: `GET /health`, `POST /predict`, and `POST /predict/batch`.
- The same WAV sample set and Locust task mix were used for the three configurations.

The API containers each ran one Uvicorn process. Nginx distributed incoming requests across the configured API containers.

## Aggregate results

| Metric | 1 container | 2 containers | 5 containers |
|---|---:|---:|---:|
| Total requests | 176 | 179 | 176 |
| Failed requests | 0 | 0 | 0 |
| Median response time (ms) | 700 | 1,200 | 1,200 |
| Average response time (ms) | 1,368.75 | 1,313.87 | 1,320.87 |
| Minimum response time (ms) | 366.28 | 363.83 | 362.89 |
| Maximum response time (ms) | 6,005.50 | 4,350.78 | 4,339.84 |
| 95th percentile (ms) | 4,600 | 4,000 | 4,000 |
| Throughput (requests/second) | 1.4756 | 1.4946 | 1.4754 |

## Endpoint-level average response times

| Endpoint | 1 container (ms) | 2 containers (ms) | 5 containers (ms) |
|---|---:|---:|---:|
| `GET /health` | 581.82 | 480.45 | 479.41 |
| `POST /predict` | 1,712.13 | 1,433.50 | 1,542.64 |
| `POST /predict/batch` | 4,673.01 | 3,817.36 | 3,876.43 |

## Interpretation

All three configurations completed the test without failed requests. The two-container configuration produced the highest measured throughput (approximately 1.495 requests per second) and the lowest aggregate average response time (approximately 1.314 seconds). It also had the lowest average response time for both single-file and batch prediction.

Increasing the replica count from two to five did not improve throughput. The five-container configuration processed approximately 1.475 requests per second, almost the same as the one-container configuration, and slightly below the two-container configuration. Its average response time was also marginally higher than the two-container result.

Batch prediction was consistently the slowest endpoint, with average response times between approximately 3.82 and 4.67 seconds. This indicates that batch inference was the most time-consuming operation in this workload.

For the conditions tested, two containers provided the best observed balance of response time and throughput. The results show that adding replicas does not automatically improve performance when the request load is relatively small or when another resource or application bottleneck limits throughput.

## Limitations

- Each run lasted only two minutes and used five concurrent Locust users. These results do not establish the maximum capacity of the service.
- The total request counts and endpoint mix varied slightly between runs, so the measurements are not perfectly controlled repetitions.
- The Killercoda environment is a shared, constrained playground. CPU scheduling, network forwarding, and resource contention may affect results.
- The aggregate median is influenced by the mix of fast health checks and slower prediction requests; endpoint-level averages provide additional context.
- The experiment measured replicas behind Nginx in Killercoda. It is separate from the Render load test, which varied the number of Locust users but did not vary the number of API containers.
- These measurements describe this test setup only and should not be presented as production capacity or a guarantee of scaling behaviour.

## Conclusion

The two-container configuration performed best in this experiment. It achieved the highest throughput and the lowest average response time, while all configurations recorded zero failures. Increasing the number of API containers to five did not provide additional performance gains under the tested load. A stronger scalability assessment would require longer tests, repeated runs, higher concurrency levels, and detailed CPU and memory monitoring.

## Evidence files

The Locust statistics were captured for each configuration. Preserve the original Locust CSV exports alongside this report if they are available. The summary values in this document are transcribed from the Locust statistics supplied for the three runs.
