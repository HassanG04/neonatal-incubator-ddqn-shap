# Related work and citation corrections

Sources checked on 8 November 2026. These studies motivate the methods and evaluation questions. Their patients, endpoints and controllers differ from this educational simulator, so numerical results are not directly comparable.

| Study | Method and evidence | Limitation for this project | Project contribution |
|---|---|---|---|
| Kaltsogianni et al., Acta Paediatrica 114 (2025), 1222-1228 | Randomized automated versus manual oxygen control in 40 ventilated late-preterm/term infants. More time in the study's 92-96% target range and fewer manual adjustments; no significant difference in supplemental-oxygen duration. | Different population and target band; oxygen control only. It does not validate this simulator or DDQN. | Joint heater/oxygen action analysis in a reproducible synthetic benchmark. |
| van Hasselt, Guez and Silver, AAAI (2016), foundational | Double DQN separates action selection from target evaluation and evaluates on Atari games. | Game benchmarks provide no neonatal safety evidence. | Applies the DDQN update to 25 joint simulated actuator settings. |
| Lundberg and Lee, NeurIPS (2017), foundational | SHAP unifies additive feature attribution methods. | Attribution depends on the reference distribution and is not causal evidence. | Fixed-head, selected-head and contrastive Q explanations with numerical replay. |
| SUPPORT Study Group, NEJM (2010), foundational | Randomized oxygen saturation target comparison, 85-89% versus 91-95%, in 1,316 extremely preterm infants. | This trial does not prescribe the project's 85/96% termination thresholds. | Explicit distinction between target-band reward, modeled limits and patient outcomes. |

## Verified references

1. Kaltsogianni O, et al. Closed-loop automated oxygen control in late preterm and term, ventilated infants: A randomised controlled trial. Acta Paediatr. 2025;114:1222-1228. First online 14 December 2024. https://doi.org/10.1111/apa.17549
2. van Hasselt H, Guez A, Silver D. Deep Reinforcement Learning with Double Q-learning. AAAI 2016. https://arxiv.org/abs/1509.06461
3. Lundberg SM, Lee SI. A Unified Approach to Interpreting Model Predictions. NeurIPS 2017. https://arxiv.org/abs/1705.07874
4. SUPPORT Study Group. Target Ranges of Oxygen Saturation in Extremely Preterm Infants. N Engl J Med. 2010;362:1959-1969. https://doi.org/10.1056/NEJMoa0911781
5. NICE. Specialist neonatal respiratory care for babies born preterm, NG124, recommendation 1.4.2. After initial stabilization, the guidance targets oxygen saturation of 91-95% in preterm babies. https://www.nice.org.uk/guidance/ng124/chapter/Recommendations#monitoring

## Scope of the correction

The former reference DOI 10.1056/NEJMoa0911783 and PMID 20472939 identify *Early CPAP versus Surfactant in Extremely Preterm Infants*. That article is not the oxygen-targeting paper. The simulator's 85% and 96% termination limits are educational modeling choices; neither the oxygen-targeting trial nor NICE establishes those exact limits as universal clinical abort thresholds. The 91-95% reward target has a specific guidance source, which does not validate the other equations or parameters. No parameter, model weight or recorded metric was changed by this correction.
