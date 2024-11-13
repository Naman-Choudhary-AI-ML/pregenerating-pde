Nx1 = 41; Rx1 = 1.00;
Nx2 = 41; Rx2 = 1.00;
Nx3 = 41; Rx3 = 1.00;
Ny = 41; Ry = 1.00;
Nb = 41; Rb = 1.00;
Nc = 41; Rc = 1.00;

Point(1) = {-15, -7.5, 0, 1.0};
//+
Point(2) = {-7.5, -7.5, 0, 1.0};
//+
Point(3) = {7.5, -7.5, 0, 1.0};
//+
Point(4) = {15, -7.5, 0, 1.0};
//+
Point(5) = {15, 7.5, 0, 1.0};
//+
Point(6) = {7.5, 7.5, 0, 1.0};
//+
Point(7) = {-7.5, 7.5, 0, 1.0};
//+
Point(8) = {-15, 7.5, 0, 1.0};
//+
Point(9) = {-0.5, 0.5, 0, 1.0};
//+
Point(10) = {0.5, 0.5, 0, 1.0};
//+
Point(11) = {0.5, -0.5, 0, 1.0};
//+
Point(12) = {-0.5, -0.5, 0, 1.0};
//+
Line(1) = {1, 2}; Transfinite Curve {1} = Nx1 Using Progression Rx1;
//+
Line(2) = {2, 3}; Transfinite Curve {2} = Nx2 Using Progression Rx2;
//+
Line(3) = {3, 4}; Transfinite Curve {3} = Nx3 Using Progression Rx3;
//+
Line(4) = {4, 5}; Transfinite Curve {4} = Nx1 Using Progression Rx1;
//+
Line(5) = {5, 6}; Transfinite Curve {5} = Nx2 Using Progression Rx2;
//+
Line(6) = {6, 7}; Transfinite Curve {6} = Nx3 Using Progression Rx3;
//+
Line(7) = {7, 8}; Transfinite Curve {7} = Ny Using Progression Ry;
//+
Line(8) = {8, 1}; Transfinite Curve {8} = Ny Using Progression Ry;
//+
Line(9) = {12, 11}; Transfinite Curve {9} = Ny Using Progression Ry;
//+
Line(10) = {11, 10}; Transfinite Curve {10} = Ny Using Progression Ry;
//+
Line(11) = {10, 9}; Transfinite Curve {11} = Nc Using Progression Rc;
//+
Line(12) = {9, 12}; Transfinite Curve {12} = Nc Using Progression Rc;
//+
Line(13) = {2, 7}; Transfinite Curve {13} = Nc Using Progression Rc;
//+
Line(14) = {3, 6}; Transfinite Curve {14} = Nc Using Progression Rc;
//+
Line(15) = {2, 12}; Transfinite Curve {15} = Nb Using Progression Rb;
//+
Line(16) = {3, 11}; Transfinite Curve {16} = Nb Using Progression Rb;
//+
Line(17) = {6, 10}; Transfinite Curve {17} = Nb Using Progression Rb;
//+
Line(18) = {7, 9}; Transfinite Curve {18} = Nb Using Progression Rb;
//+
Curve Loop(1) = {15, 9, -16, -2};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {16, 10, -17, -14};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {17, 11, -18, -6};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {18, 12, -15, 13};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {8, 1, 13, 7};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {4, 5, -14, 3};
//+
Plane Surface(6) = {6};
//+
Transfinite Surface {1};
//+
Transfinite Surface {2};
//+
Transfinite Surface {3};
//+
Transfinite Surface {4};
//+
Transfinite Surface {5};
//+
Transfinite Surface {6};
//+
Recombine Surface {1};
//+
Recombine Surface {2};
//+
Recombine Surface {3};
//+
Recombine Surface {4};
//+
Recombine Surface {5};
//+
Recombine Surface {6};
//+
Extrude {0, 0, 1} {
  Surface{1, 2, 3, 4, 5, 6};
  Layers{1};
  Recombine; 
}
//+
Physical Surface("Inlet", 151) = {115};
//+
Physical Surface("Outlet", 152) = {137};
//+
Physical Surface("Top", 153) = {127, 83, 141};
//+
Physical Surface("Bottom", 154) = {119, 39, 149};
//+
Physical Surface("Cylinder", 156) = {75, 53, 31, 97};
//+
Physical Surface("FrontBack", 155) = {128, 40, 62, 84, 106, 150, 6, 1, 4, 3, 2, 5};
//+
Physical Volume("internal", 157) = {1, 2, 3, 4, 5, 6};
