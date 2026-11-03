Observations: <what you see in the weights>
- W1 pixel rows for outer frame are ~0. All of rows 0..11, i.e. pixel 0..11, top image row, are e.g.
`0.001 0.006 -0.006 0.005 0.012 0.015 0.012 ...`
`-0.001 0.001 -0.002 0.008 0.006 -0.004 0.001 ...` max |w| ~0.02-0.03.
Same for col 0 / col 11 through the image, e.g. p12:
`0.004 0.003 -0.012 -0.002 0.013 ...`, p23 `0.011 -0.004 ...`, p36 `-0.009 ...`, etc. max ~0.02.
Last pixel rows are same, e.g. `0.005 -0.012 -0.004 ...`, `0.011 -0.001 0.007 ...` max ~0.02.

- W1 pixel rows for central area rows 1..10 cols 1..10 are 10-20x larger, ~0.1-0.5, localized. E.g. p15-p18, r1 c3-6:
`-0.003 ... -0.122 ... 0.112 ...`
`0.037 0.041 0.054 -0.142 0.059 ... 0.200 ... 0.178 ... -0.134 ...`
`0.083 0.091 0.137 -0.033 0.026 ... 0.402 ... 0.162 ... 0.141 ...`
`0.046 0.099 0.136 0.047 ... 0.437 ... 0.209 ...`
`0.016 0.080 0.081 ... 0.339 ... 0.251 ... 0.160 ...`
etc. Similarly r2 c2-9 e.g.
`0.057 0.058 0.047 ... 0.166 0.182 ... 0.221 ...`
`0.163 0.133 0.156 0.144 ... 0.365 ... 0.189 ...`
`0.153 0.179 0.176 ... 0.244 ... 0.235 ...`
and r3-r7 e.g.
`0.001 0.035 ... -0.165 ... 0.272 ...`,
`0.206 0.117 0.124 ... 0.244 ... 0.310 ... -0.317 ...`,
`0.317 0.161 0.164 ... 0.235 ... -0.392 ...`,
`0.413 ... 0.178 ... 0.259 ... -0.319 ...`,
`0.381 ... 0.241 ... 0.248 ... -0.350 ...`,
`0.324 ... 0.448 ... 0.339 ... -0.202 ...`,
etc. Each hidden unit 0..63 pools only a few adjacent pixels.

- b3 is ~-0.5 on frame, ~-0.1 to +0.18 in centre. E.g. top row:
`-0.551 -0.602 -0.555 -0.453 -0.329 -0.344 -0.380 -0.439 -0.448 -0.465 -0.523 -0.605`
left col: `-0.551 -0.444 -0.457 -0.554 -0.471 -0.435 -0.435 -0.345 -0.501 -0.513 -0.599 -0.476`
right col: `-0.605 -0.579 -0.483 -0.588 -0.519 -0.357 -0.426 -0.341 -0.492 -0.509 -0.586 -0.571`
bottom row: `-0.476 -0.489 -0.571 -0.518 -0.372 -0.281 -0.388 -0.367 -0.381 -0.534 -0.428 -0.571`
centre e.g. r2: `-0.225 -0.163 -0.101 0.029 0.012 0.003`, r4: `-0.097 0.026 -0.011 -0.015 0.056 0.024 0.106`, r5: `-0.131 0.011 0.081 -0.001 0.057 0.182 0.122 0.076`, r6: `-0.064 -0.024 0.087 0.037 0.115 0.094 0.006 0.098`, etc.

- W3 is ~0 for frame pixels, large localized positive blobs for centre pixels. E.g. row0:
`-0.002 -0.002 -0.003 -0.005 -0.015 -0.021 -0.008 0.007 ...` for pixels 0..11, and `... 0.001 0.009 0.005 -0.008 -0.008 -0.002 -0.001 -0.001 -0.002 -0.002` for pixels 132..143.
Centre e.g.:
`... 0.008 0.077 0.295 0.467 0.500 0.393 0.079 ...` — 3-4 adjacent hori­zontal pixels in same image row ~0.3-0.5, and 12 numbers later, i.e. same column next row down: `... 0.081 0.145 0.227 ...`, i.e. vertical alignment.
Another row: `... 0.095 0.538 -0.106 ...` and 12 later `... 0.039 0.515 0.226 ...`:
`0.095 0.538` upper row, `0.039 0.515 0.226` lower row, distance exactly 12 = one image row. I.e. compact 2-D ~3x3 patch.
Many rows show this: `... 0.085 0.352 ...`, `... 0.095 0.538 ... 0.515 0.226 ... 0.389 ...`, `... 0.244 0.419 0.525 0.582 0.367 ...`, etc. Each of 64 rows paints a different centre location.

H=64 = 8x8.

Reasoning: <why>
Border pixels have no W1 input weight and b3 ~ -0.55 and no W3 weight: they are always black -1. Network does not need to read or write them.
Centre 10x10 pixels have large W1 read weights and large W3 write weights and b3 ~0: they sometimes white +1 sometimes black -1.
W1 columns and W3 rows are spatially localized: a hidden unit reads a few neighbouring noisy pixels and writes a few neighbouring clean pixels at same 12x12 location. Different units tile the centre.
That is exactly denoising of a small compact white object on black: to denoise you average the noisy input over the object footprint, and to generate you add white at that footprint.
Number of tilings 64 = 8x8 positions of a 3x3 block in a 10x10 interior with 1-pixel black frame all around: top-left can be rows 1..8, cols 1..8. Patch width seen in W3 is 3 strong horizontals e.g. `0.467 0.500 0.393`, height 2-3 vertical repeats 12 apart. No evidence for hollow, elongated, multi-object, centred-only shape: average bias is flat grey over centre, patches occur everywhere in centre from r1 to r10, c1 to c10.
Ruled out: full-image textures, centered large circle/square, lines/bars spanning image, MNIST-like strokes, random single pixels over whole image — all would need large weights on frame and/or non-localized stripe weights, not seen.

Images: <what>
Black 12x12, -1 background with 1-pixel permanent black frame, containing one single solid ~3x3 white +1 square, randomly located in the interior. I.e. 3 rows x 3 cols of white, rest black, top-left corner uniformly among the 64 positions rows 1-8, cols 1-8. Generation varies only position, not size/shape/colour.

Example: <12x12, # white, . black>
```
............
............
............
............
....###.....
....###.....
....###.....
............
............
............
............
............
```
Any translation with 1-pixel margin is typical.

Confidence: 85%
