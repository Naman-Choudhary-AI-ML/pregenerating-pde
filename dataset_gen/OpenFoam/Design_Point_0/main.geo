//+
Nx1 = 41; Rx1 = 1.00;
Nx2 = 41; Rx2 = 1.00;
Nx3 = 41; Rx3 = 1.00;
Ny = 41; Ry = 1.00;
Nb = 41; Rb = 1.00;
Nc = 41; Rc = 1.00;

Point(1) = {-15, -7.5, 0, 1.0};
Point(2) = {-7.5, -7.5, 0, 1.0};
Point(3) = {7.5, -7.5, 0, 1.0};
Point(4) = {15, -7.5, 0, 1.0};
Point(5) = {-15, 7.5, 0, 1.0};
Point(6) = {-7.5, 7.5, 0, 1.0};
Point(7) = {7.5, 7.5, 0, 1.0};
Point(8) = {15, 7.5, 0, 1.0};

//Cylinder Points
Point(9) = {-0.35355339, -0.35355339, 0, 1.0};
Point(10) = {0.35355339, -0.35355339, 0, 1.0};
Point(11) = {-0.35355339, 0.35355339, 0, 1.0};
Point(12) = {0.35355339, 0.35355339, 0, 1.0};
Point(13) = {0, 0, 0, 1.0};

Line(1) = {1, 2}; Transfinite Curve {1} = Nx1 Using Progression Rx1;
//+
Line(2) = {2, 3}; Transfinite Curve {2} = Nx2 Using Progression Rx2;
//+
Line(3) = {3, 4}; Transfinite Curve {3} = Nx3 Using Progression Rx3;
//+
Line(4) = {5, 6}; Transfinite Curve {4} = Nx1 Using Progression Rx1;
//+
Line(5) = {6, 7}; Transfinite Curve {5} = Nx2 Using Progression Rx2;
//+
Line(6) = {7, 8}; Transfinite Curve {6} = Nx3 Using Progression Rx3;
//+
Line(7) = {1, 5}; Transfinite Curve {7} = Ny Using Progression Ry;
//+
Line(8) = {2, 6}; Transfinite Curve {8} = Ny Using Progression Ry;
//+
Line(9) = {3, 7}; Transfinite Curve {9} = Ny Using Progression Ry;
//+
Line(10) = {4, 8}; Transfinite Curve {10} = Ny Using Progression Ry;

//Cylinder Lines
Circle(11) = {9, 13, 10}; Transfinite Curve {11} = Nc Using Progression Rc;
//+
Circle(12) = {10, 13, 12}; Transfinite Curve {12} = Nc Using Progression Rc;
//
Circle(13) = {12, 13, 11}; Transfinite Curve {13} = Nc Using Progression Rc;
//+
Circle(14) = {11, 13, 9}; Transfinite Curve {14} = Nc Using Progression Rc;

//Block Lines
Line(15) = {2, 9}; Transfinite Curve {15} = Nb Using Progression Rb;
//+
Line(16) = {3, 10}; Transfinite Curve {16} = Nb Using Progression Rb;
//+
Line(17) = {7, 12}; Transfinite Curve {17} = Nb Using Progression Rb;
//+
Line(18) = {6, 11}; Transfinite Curve {18} = Nb Using Progression Rb;

//Surfaces
Curve Loop(1) = {15, 11, -16, -2};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {16, 12, -17, -9};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {17, 13, -18, 5};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {18, 14, -15, 8};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {4, -8, -1, 7};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {6, -10, -3, 9};
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
Recombine Surface {2};
Recombine Surface {3};
Recombine Surface {4};
Recombine Surface {5};
Recombine Surface {6};
//+
Extrude {0, 0, 1} {
  Surface{1,2,3,4,5,6};
  Layers{1};
  Recombine;
}
//+
Physical Surface("Inlet", 151) = {127};
//+
Physical Surface("Outlet", 152) = {141};
//+
Physical Surface("Top", 153) = {115, 83, 137};
//+
Physical Surface("Bottom", 154) = {123, 39, 145};
//+
Physical Surface("Cylinder", 155) = {75, 97, 31, 53};
//+
Physical Surface("FrontBack", 156) = {40, 62, 84, 106, 128, 150, 1, 4, 3, 2, 5, 6};
//+
Physical Volume("internal", 157) = {1, 2, 3, 4, 5, 6};
