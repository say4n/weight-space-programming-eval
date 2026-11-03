The car starts at x~2.8 and must reach the blue box at x~0.23. Between them are a thin red strip on the floor at x 1.18-2.065, y~0.4, fatal on green contact, and a slanted static block at x~2.2, y~0.3-0.82.

The only ground route left must mount the block and jump the red: green must be airborne while x is over 1.18-2.065. The sample shows this working in part: +1 = left drive moves the car 2.8 -> 2.38 -> 2.0, green to 1.75,1.0 in the air over the red, other wheel at 2.3 still right of red, body high-centered on the block.

Then it sticks there from step 32 to 256. The entry-0 probabilities flip-flop: 1.00/0/0 then 0.07/0.93/0 then 1.00/0/0 etc. i.e. motor off vs drive +1 alternating. Torque is cut exactly when contact push is needed, momentum is lost, the car never gets over the block. Result 256 timeouts, progress 0.339 = distance to the stuck point, small shaping return 0.152, 0 red because green stays in the air, 0 success.

The head logits for entry 0 are small state-dependent values, hence indecisive. Entries 1-5 drive nothing. The fix is to latch entry 0 to +1, continuous left drive.

```edits
actor_head.W[:, 0:3] = 0
actor_head.b[0:3] = [-10, 10, -10]
```

This zeroes only the 3 logits for motor 0, removing state dependence, and biases strongly to value 1. Logits become -10,+10,-10 so softmax is ~0/1/0 deterministically. Both wheels are driven CCW at +18 rad/s, i.e. left, all the time. This preserves the learned left direction but removes the off-bursts that cause high-centering. Continuous torque lets the car climb the block, stay airborne over the red with green, land left of x~1.18 and continue left to touch blue. Other heads are untouched and have no effect.