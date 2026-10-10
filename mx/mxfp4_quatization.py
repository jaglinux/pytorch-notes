#mfp4 possible range
0
±0.5
±1
±1.5
±2
±3
±4
±6

# For each block of 32 values

max_abs = max(abs(x))
scale = nearest_power_of_2(max_abs / 6)
scaled = x / scale
clipped = clip(scaled, -6, 6)
mxfp4 = round_to_E2M1(clipped)

# Quantization example

x = [1.2, 3.7, 10.5, -15.0]
max_abs = 15
# mx fp4 range
[-6, 6]
#scale
15 / 6 = 2.5
# but for mx, scale is nearest suitable power-of-two scale
scale = 4 
# but stored as (Stored as exponent 2 in E8M0 since 4 = 2².)

# now calculate scaled values
1.2   / 4 = 0.30
3.7   / 4 = 0.925
10.5  / 4 = 2.625
-15.0 / 4 = -3.75

# clip it , in this example no changes.
# then round it to nearest E2M1 value 
[0.5, 1.0, 3.0, -4.0]

# De Quantization example
#Multiply each FP4 value by the block scale, here scale is 4
0.5 × 4 = 2
1.0 × 4 = 4
3.0 × 4 = 12
-4.0 × 4 = -16
# Recovered tensor
[2, 4, 12, -16]
