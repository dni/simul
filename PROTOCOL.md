# Simul device protocol v0.2

Newline-delimited JSON over serial (USB CDC, 115200 baud). One request line, one reply line.
All values are SI units: volts, amperes, seconds, kelvin, tesla, pascal; humidity as a fraction.
Every reply has `"ok": true`, or `"ok": false, "error": "<message>"`.

Channel names on the device must match the source/sensor names in the `rig` block, and the
rig's `device` line must match the name the device reports. That is how a rig description is
bound to one physical board.

| request | reply |
|---|---|
| `{"op":"hello"}` | `{"ok":true,"device":"pico_rc_1","fw":"0.2","channels":["V","Vc"]}` |
| `{"op":"describe"}` | `{"ok":true,"channels":[{"name":"Vc","role":"sensor","kind":"voltage","where":"across","target":"C","range":[0,3.3],"accuracy":[0.01,0.015],"module":"adc0","calibrated":"2026-10-01"}]}` |
| `{"op":"set","ch":"V","value":1.5,"settle_ms":2000}` | `{"ok":true}` (returns after settling) |
| `{"op":"read","ch":["Vr","Ir"],"n":16}` | `{"ok":true,"values":{"Vr":1.49,"Ir":0.00148}}` (mean of n) |
| `{"op":"step","ch":"V","lo":0,"hi":3.3,"sample":["Vc"],"dt":2e-5,"n":250}` | `{"ok":true,"data":{"t":[...],"Vc":[...]}}` |

`describe` reports each channel's own descriptor. `accuracy` is `[relative, absolute]`, a
worst-case bound: |reading − true| ≤ rel·|reading| + abs. `calibrated` is the date of the
calibration that produced those numbers, or `null` when they are datasheet figures.
`simul describe serial:<port>` turns this into the instrument half of a rig file.

`step` semantics: drive `ch` to `lo` and let it settle, then switch to `hi` at t = 0 and sample
the listed channels `n` times at nominal interval `dt`. `t` holds the *actual* sample times
measured on the device. If the device cannot keep up with `dt`, it must reply with an error
rather than return mistimed data.

A device must refuse (`ok:false`) any request outside what it can physically do, for example a
GPIO source asked for 1.5 V. It must never silently approximate.
