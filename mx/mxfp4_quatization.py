# For each block of 32 values

max_abs = max(abs(x))

scale = nearest_power_of_2(max_abs / 6)

scaled = x / scale

clipped = clip(scaled, -6, 6)

mxfp4 = round_to_E2M1(clipped)
