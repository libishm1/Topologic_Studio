# The gas radiation tables

`gas-radiation-tables.f32` holds the absorption line blackbody distribution
functions of CO2 and H2O at 1 atm that the Flamescapes solver reads when its
switch `gasRadiation` is on (`src/core/gasRadiation.js`). The app fetches the
file only then.

## Source

The tables are those of

> Pearson, Webb, Solovjov and Ma (2014). Efficient
> representation of the absorption line blackbody distribution function for
> H2O, CO2, and CO at variable temperature, mole fraction, and pressure.
> Journal of Quantitative Spectroscopy and Radiative Transfer 138, 82-96.
> doi:10.1016/j.jqsrt.2014.01.019

as the RadLib library ships them:

> Stephens, Jensen, Wheeler and Lignell (2022). RadLib: a
> radiative property model library for CFD. Computer Physics Communications
> 272, 108227. doi:10.1016/j.cpc.2021.108227

RadLib, github.com/BYUignite/radlib, at commit
583e0fe63ee60dbf750ee0fca868728f50bcbaf6, files `data/rcslw_data/co2_p1_0.txt`
and `data/rcslw_data/h2o_p1_0.txt`. RadLib's note on these data says they were
taken from http://albdf.byu.edu.

SHA-256 of the two source files:

- `co2_p1_0.txt`: 5bd9c9ecdd3449f5c145a9e6bf07a181486752cec22920ef1928e9422a17508b
- `h2o_p1_0.txt`: b3dd6c860866d36c443719c3cadb2c1c1e2b83c3198040ee9d3b0847a2972e53

## What the file is

`tools/gas-radiation-tables.mjs` builds it from the two source files after
checking those SHA-256 values: their 556 640 values rounded to float32,
little-endian, with no header. First CO2's 55 664 values in the order
[T_g][T_b][C], then H2O's 500 976 in the order [Y][T_g][T_b][C], the last
index fastest; 2 226 560 bytes. No value is changed otherwise. The two text
files themselves are not in this repository.

SHA-256 of `gas-radiation-tables.f32`:
451d2cafb1f28292d5d39812aeec29246b7b8623894b21edc8c7788d66e254e6

## RadLib's licence

RadLib is distributed under the MIT licence, reproduced here in full:

```
MIT License

Copyright (c) 2020 David Lignell

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
