First, I would like to start with an interface a simple point-to-point hydraulics on a segment, bend, contraction/expansion, or bend class object for either compressible or incompressible flow (functions in @compressible_flow.py  and @incompressible.py  . Later we will add a component network solving GUI (see @network.md  for details). I would like to use pyside6 for the GUI. 

Beginning screen
-Select either Incompressible or Compressible

Pipe segment input screen
For an example of a basic input screen, I'd like it to imitate @Basic fluid mechanincs.xlsx which has input cells in green describing the line segment (OD, WT, ID, length, roughness, elevation change). Alternatively, I'd like the ability to load a CSV profile like @testprofile_ID_OD_WT.csv . Once loaded or values are entered, the GUI should display a chart with the ID vs length profile.

Fluid definition input screen
 For incompressible fluids, text boxes to enter API gravity or density and viscosity with drop down menu of common units.nd specifies the flow rate. I'd like the ability to enter flow rates in either volume rate or mass rate (or molar rate for compressible).
 For compressible fluids, I'd like to specify mole fractions of components, initial pressure, temperature, and flow rate.

After the fluid definition screen, I'd like a calculate button which, when pressed, computes the pressure drop/final pressure and displays the results as a pressure profile graph.