# Open data questions

Findings from intake, labelling and executor probes that need a human decision. Each one changes what the
system is allowed to claim.

## Jackal odometry during apparent stops (Brackenridge)
- **What the labeller saw.** The L1 labeller judged the robot "clearly stopped" at 84–86 s and 98–100 s.
- **What odometry says.** Over the same spans, both the twist and the integrated pose read about 0.50 m/s (median
  0.504; minimum 0.37).
- **What the front camera shows.** It is ambiguous: the scene is nearly static apart from a passing SUV.
- **Why it matters.** If odometry cannot report standstill, a speed-below-X clause is currently **FALSE** on this
  log ("no stop found, coverage complete") when it should be **UNKNOWN (BELOW_SENSOR_FLOOR)**. Intake measures
  `speed_floor_mps` (1st-percentile speed), but the executor does not apply it. GDC, for example, has a high
  1st percentile because Spot never stops there, not because its odometry has a floor.
- **Decision needed.** Confirm per log (visual odometry, IMU vibration or the joystick axes could settle it), then
  set the floor.

## Stale leading front-camera frames
- **Observation.** In several recordings the first two front frames (t ≈ 1.4 s) come from a different scene, followed
  by a 0.4 s gap: a buffer flush.
- **Plan.** Intake should detect this pattern and mask those ordinals.
