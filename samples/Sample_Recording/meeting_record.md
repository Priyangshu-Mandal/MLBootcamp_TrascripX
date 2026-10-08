# Meeting Record

*Source recording:* `Sample_Recording.mp4`  
*Generated:* 2026-10-08T12:35:25+00:00
*Models:* speech to text: faster-whisper large-v3-turbo (cuda, float16); refinement: openai/gpt-oss-120b (Groq); documentation: gemini-3.1-flash-lite (Gemini)

## Summary

The team reviewed project performance, confirming the FP16 model and setting a P95 latency alert threshold of 140ms. Several operational decisions were finalized, including capping cloud spend at $4,500 and rejecting both a Kafka migration and a cloud provider switch for this quarter. Action items were assigned for testing, reporting, and demo preparation, with the customer demo confirmed for October 15th.

## Minutes

### Model Performance
- P95 latency is 120ms, meeting the 150ms SLA.
- INT8 quantization is rejected for this release; the team will continue using the FP16 model.
- LoRA fine-tuning for accented speech is deferred until the next WER report.

### Deployment
- The inference service is running on Kubernetes with Terraform and QuebecTel.
- The team agreed on a 140ms P95 alert threshold for Prometheus.
- Kafka migration is rejected for this quarter.
- CI/CD rollback functionality will be added pending successful load testing.

### Budget and Demo
- Cloud spend is capped at $4,500 for the quarter.
- Switching cloud providers is rejected due to migration risk.
- The customer demo is confirmed for October 15th.

## Key Decisions

1. **Do not use INT8 quantization in this release.**
   > "No int8 in this release."
2. **Set the Prometheus alert threshold to 140ms at P95.**
   > "The alert threshold is 140 milliseconds at P95."
3. **Do not migrate the event queue to Kafka this quarter.**
   > "We are not migrating to Kafka this quarter."
4. **Cap cloud spend at $4,500 for the quarter.**
   > "Cloud Spend stays capped at $4,500."
5. **Do not switch cloud providers.**
   > "Nope. That is not approved."
6. **The customer demo is scheduled for October 15th.**
   > "The demo is October 15th."

## Action Items

| # | Task | Owner | Deadline | Status | Condition / note |
|---|------|-------|----------|--------|------------------|
| 1 | Put together a proper accented speech test set | Kabir | unspecified | confirmed |  |
| 2 | Send the latency report to Laura | Arjun | Thursday | confirmed |  |
| 3 | Add rollback functionality to CI/CD pipeline | Manav | unspecified | conditional | Only if the load test passes. |
| 4 | Run the load test | Manav | Wednesday | confirmed | Arjun's export needs to land first. |
| 5 | Configure the Prometheus alert rule | Manav | unspecified | confirmed |  |
| 6 | Update the runbook for the new Grafana dashboards | unspecified | unspecified | unassigned |  |
| 7 | Prepare the demo slides | Arjun | October 12th | confirmed |  |
| 8 | Finalize the demo script | unspecified | October 10th | unassigned |  |

### Evidence for action items

**Task 1: Put together a proper accented speech test set**
> "Kabir, can you put together a proper accented speech test set? Yes, I'll take that."

**Task 2: Send the latency report to Laura**
> "Arjun, can you send her the latency report? Sure. I will send it by Thursday."

**Task 3: Add rollback functionality to CI/CD pipeline**
> "I will add it, but only if the load test passes."

**Task 4: Run the load test**
> "Make it Wednesday. Arjun's export needs to land first, but Wednesday works."

**Task 5: Configure the Prometheus alert rule**
> "I will configure the Prometheus alert rule."

**Task 6: Update the runbook for the new Grafana dashboards**
> "Someone should also update the runbook for the new Grafana dashboards."

**Task 7: Prepare the demo slides**
> "Arjun, please prepare the demo slides by October 12th. Okay. Will do."

**Task 8: Finalize the demo script**
> "And somebody needs to finalize the demo script by October 10th."
