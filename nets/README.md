This is the folder for task-specific networks.


## Darcy Flow 

### Encoder 

E1:
- original version
- graph-based cross attention, do not support different position

E2:
- vector-based cross attention (first qk, then v), support different position
- **performance drop**

E3:
- vector-based cross attention (first qk, then v), support different position
- **performance drop**

E4:
- move cross attention to Decoder
- **recover performance**



### Decoder

D1:
- original version

D2:
SKIP

D3:
SKIP

D4:
- add cross attention module from encoder
- **recover performance**

D5:
- Directly apply the decoder from OFormer
- **performance drop**

D6:
- remove propagator for Darcy Flow

D6a:
- Update Decoder decode() by concatenating x & propagate_pos

D6b:
- Rotary PE

D6c:
- Update Decoder decode() by add the lap_pos_enc_layer

### Combination

- M1: E1 + D1
- M2: E2 + D1
- M3: E3 + D1
- M4: E4 + D4
- E4D5: E4 + D5
- M6: D4 + D6
