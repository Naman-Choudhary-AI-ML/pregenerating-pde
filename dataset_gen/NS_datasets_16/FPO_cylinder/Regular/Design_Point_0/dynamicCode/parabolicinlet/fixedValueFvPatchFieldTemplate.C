/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2019-2021 OpenCFD Ltd.
    Copyright (C) YEAR AUTHOR, AFFILIATION
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "fixedValueFvPatchFieldTemplate.H"
#include "addToRunTimeSelectionTable.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "unitConversion.H"
#include "PatchFunction1.H"

//{{{ begin codeInclude

//}}} end codeInclude


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * Local Functions * * * * * * * * * * * * * * //

//{{{ begin localCode

//}}} end localCode


// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //

// dynamicCode:
// SHA1 = 4687328d2566dd4dd352af5a88e623a74e1ea6be
//
// unique function name that can be checked if the correct library version
// has been loaded
extern "C" void parabolicinlet_4687328d2566dd4dd352af5a88e623a74e1ea6be(bool load)
{
    if (load)
    {
        // Code that can be explicitly executed after loading
    }
    else
    {
        // Code that can be explicitly executed before unloading
    }
}

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

makeRemovablePatchTypeField
(
    fvPatchVectorField,
    parabolicinletFixedValueFvPatchVectorField
);

} // End namespace Foam


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::
parabolicinletFixedValueFvPatchVectorField::
parabolicinletFixedValueFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF
)
:
    parent_bctype(p, iF)
{
    if (false)
    {
        printMessage("Construct parabolicinlet : patch/DimensionedField");
    }
}


Foam::
parabolicinletFixedValueFvPatchVectorField::
parabolicinletFixedValueFvPatchVectorField
(
    const parabolicinletFixedValueFvPatchVectorField& rhs,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    parent_bctype(rhs, p, iF, mapper)
{
    if (false)
    {
        printMessage("Construct parabolicinlet : patch/DimensionedField/mapper");
    }
}


Foam::
parabolicinletFixedValueFvPatchVectorField::
parabolicinletFixedValueFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict
)
:
    parent_bctype(p, iF, dict)
{
    if (false)
    {
        printMessage("Construct parabolicinlet : patch/dictionary");
    }
}


Foam::
parabolicinletFixedValueFvPatchVectorField::
parabolicinletFixedValueFvPatchVectorField
(
    const parabolicinletFixedValueFvPatchVectorField& rhs
)
:
    parent_bctype(rhs),
    dictionaryContent(rhs)
{
    if (false)
    {
        printMessage("Copy construct parabolicinlet");
    }
}


Foam::
parabolicinletFixedValueFvPatchVectorField::
parabolicinletFixedValueFvPatchVectorField
(
    const parabolicinletFixedValueFvPatchVectorField& rhs,
    const DimensionedField<vector, volMesh>& iF
)
:
    parent_bctype(rhs, iF)
{
    if (false)
    {
        printMessage("Construct parabolicinlet : copy/DimensionedField");
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::
parabolicinletFixedValueFvPatchVectorField::
~parabolicinletFixedValueFvPatchVectorField()
{
    if (false)
    {
        printMessage("Destroy parabolicinlet");
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void
Foam::
parabolicinletFixedValueFvPatchVectorField::updateCoeffs()
{
    if (this->updated())
    {
        return;
    }

    if (false)
    {
        printMessage("updateCoeffs parabolicinlet");
    }

//{{{ begin code
    #line 20 "/home/namancho/openfoam/openfoam_project/Flowbench_Openfoam/FPO_cylinder/Regular/Design_Point_0/0/U/boundaryField/left"
// Channel height
            const scalar H = 64.0;
            // Maximum velocity to get Re=100 if nu=0.01 and L=1 (Re=Umax*L/nu)
            const scalar Umax = 1.0;
            // Get patch cell center positions (only need the y-coordinate)
            const vectorField& pos = patch().Cf();
            vectorField Uvec(pos.size());
            forAll(pos, i)
            {
                scalar y = pos[i].y();
                // Parabolic profile: zero at y=0 and y=H, max at y=H/2.
                scalar u = 4.0 * Umax * y * (H - y) / (H * H);
                Uvec[i] = vector(u, 0.0, 0.0);
            }
            operator==(Uvec);
//}}} end code

    this->parent_bctype::updateCoeffs();
}


// ************************************************************************* //

