# Abstention controls

Questions whose **true answer is "nowhere"**, and which the system can prove. The best answer is
`none_found_exhaustive` with no results (`proved_none`). Returning nothing with `insufficient_evidence` is an
honest abstention without proof (`abstained`). Returning ranked results is a failure to abstain, however plausible
they look. These measure the ability to say "nowhere, it's tight" from the data.

## ctl_odom_gyro_agree: wheel odometry against the gyro (Jackal)

The question asks where wheel odometry and the gyro disagree about the turn rate by **more than 30 °/s for at least
0.5 s**. The feature is `odom_gyro_yaw_disagreement_dps` (provider imu@2): |odometry yaw rate − gyro yaw rate|,
both as trailing 0.5 s time-means, computed on every IMU sample (70 Hz).

| Recording | median | p99 | max (instant) | max held 0.5 s |
|---|---|---|---|---|
| Brackenridge | 0.7 | 5.5 | 9.6 | 3.6 |
| Sanjac | 1.1 | 13.9 | 27.8 | 10.6 |
| Library_Fountain_5 | 1.0 | 4.7 | 8.2 | 4.2 |
| Sanjac_Rec_91 | 1.1 | 7.9 | 13.9 | 5.0 |
| Stadium_Sanjac_90 | 1.4 | 16.2 | 26.4 | 6.1 |

All values are °/s. No single instant crosses 30 °/s, and no half-second crosses 11 °/s, so the answer is nowhere
with about a 3× margin. The two signals agree at r = 0.985–0.992 with a scale of 0.96–1.04. Spot records no IMU, so
the question is scoped to Jackal. On Spot the clause would be UNKNOWN (sensor absent), never FALSE.

Why it is a good control: an earlier analysis compared a 0.5 s odometry average with a 0.2 s gyro average during a
fast turn, and wrongly reported a wheel slip (odometry −44 °/s, gyro +4 °/s). The raw samples show the same turn in
both. The trap is plausible, and the data is unambiguous.
