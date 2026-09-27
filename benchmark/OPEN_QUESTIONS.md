# Open data questions

Findings from intake, labelling and executor probes that need a human decision. Each one changes what the
system is allowed to claim.

## Jackal odometry during apparent stops (Brackenridge): resolved, the robot was moving
- **What the labeller saw.** The L1 labeller judged the robot stopped at 84–86 s and 98–100 s. The captions for
  `Brackenridge:0084`, `:0088` and `:0100` say "stops", "waits" and "pauses".
- **What three sensors say.** Odometry reads 0.50 m/s over those spans.
  - **Joystick.** The operator holds the forward stick at full scale for the whole recording. Button 4 is held
    throughout; button 5 toggles speed. Odometry reads exactly 0.50 m/s whenever button 5 is released and about
    2.0 m/s whenever it is held. That matches the teleop joystick's normal and turbo scales, with turbo at the
    profile's maximum. Both "stops" are the operator releasing turbo.
  - **IMU vibration.** 0.17–0.33 m/s² there, against 0.8–1.1 at 2 m/s and under 0.03 at the true standstills in
    Sanjac and Sanjac_Rec_91. This is slow driving, not a stop.
- **Conclusion.** Jackal odometry does report standstill. Brackenridge never stops after it starts, so its 1st
  percentile speed (0.27 m/s) is real. The three L1 captions are wrong about stopping, probably because the front
  camera looks nearly static at 0.5 m/s.
- **Corrected.** The `l1-corrections-2026-09-27` campaign (priority 1; `l1-original` is left untouched) re-labelled
  the three segments from the data. The judge confirmed from odometry that the robot never stops (its floor is
  0.5 m/s, with brief dips to about 0.37). The new captions describe moving at 0.5–2 m/s. The IMU eval suite
  (`providers/imu`) pins the case.

## Topics recorded without a message definition (Jackal)
- **Observation.** SCAND's Jackal bags record three topics with an empty message definition: `/imu/data_raw`,
  `/status` and (in Library_Fountain_5) `/navsat/nmea_sentence`. Intake's schema check records them.
- **IMU.** Its recorded md5 matches the standard `sensor_msgs/Imu`, so the decoder uses that definition. Every
  message carries 1 surplus trailing byte, verified identical on 50 sampled messages per recording. The values are
  physical: |a| ≈ 9.8 m/s². Gyro and wheel-odometry yaw rate, compared on the same 0.5 s window, agree at
  r = 0.985–0.992 with a scale of 0.96–1.04. No sustained disagreement above 20°/s for 0.2 s occurs in any Jackal
  log, so there is no wheel slip to find. (An earlier "−44 vs +4°/s" and "slope 0.9" came from comparing a 0.5 s
  odometry average with a 0.2 s gyro average during a fast-changing turn.)
- **Undecodable.** `/status` (`jackal_msgs/Status`, which normally carries battery voltage) and
  `nmea_msgs/Sentence` are not standard types. A definition could be restored only if its md5 matches the recorded
  one. Until then, "battery is not recorded" holds for Spot, while for Jackal it is recorded but undecodable.

## Stale leading front-camera frames
- **Observation.** In several recordings the first two front frames (t ≈ 1.4 s) come from a different scene, followed
  by a 0.4 s gap: a buffer flush.
- **Status.** Resolved for derived data. Intake records `stale_leading_frames` per topic (Butler 2, GDC 2, Rec_Tent_129 1;
  none on Jackal), and detections@2 and the window index skip those ordinals. Raw clips still show them, since they
  are in the recording.

## Front camera absent in Bass_Garage_134
- **Observation.** This SCAND Val-split Spot log has every Spot topic except `/image_raw/compressed`.
- **Status.** Intake recognises it as Spot from its identity topics and records `absent_sensors: [front_camera]`.
  Front-camera clauses evaluate to `UNKNOWN(SENSOR_ABSENT_IN_LOG)`, detections are not applicable, and its windows
  use front-stereo body-camera vectors (`source=body_cameras`). Its eval rows are reported separately.
