import numpy as np

# Number of quadrants: 2x2
# Quadrants:
# Q1: x in [0, 0.5), y in [0, 0.5)
# Q2: x in [0.5, 1.0), y in [0, 0.5)
# Q3: x in [0, 0.5), y in [0.5, 1.0)
# Q4: x in [0.5, 1.0), y in [0.5, 1.0)

# Distributions for initial conditions:
# rho  ~ U[0.1, 1]
# vx   ~ U[-1, 1]
# vy   ~ U[-1, 1]
# p    ~ U[0.1, 1]

def sample_uniform(low, high):
    return np.random.uniform(low, high)

# Sample for each quadrant
quadrants = {}
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    rho = sample_uniform(0.1, 1.0)
    vx  = sample_uniform(-1.0, 1.0)
    vy  = sample_uniform(-1.0, 1.0)
    p   = sample_uniform(0.1, 1.0)
    
    # Assume ideal gas with R=1 for simplicity (dimensionless)
    R = 287
    T = p/(rho*R)
    
    quadrants[q] = {
        'rho': rho,
        'vx': vx,
        'vy': vy,
        'p' : p,
        'T' : T
    }

# Now create the setFieldsDict
setFieldsDict_content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      setFieldsDict;
}
defaultFieldValues
(
    volVectorFieldValue U (0 0 0)
    volScalarFieldValue p 1
    volScalarFieldValue T 1
);

regions
(
"""
# Define each quadrant region and its field values
# Quadrant 1: [0,0.5)x[0,0.5)
setFieldsDict_content += f"""
    boxToCell
    {{
        type    boxToCell;
        box     (0 0 0) (0.5 0.5 0.5);
        fieldValues
        (
            volVectorFieldValue U ({quadrants['Q1']['vx']} {quadrants['Q1']['vy']} 0)
            volScalarFieldValue p {quadrants['Q1']['p']}
            volScalarFieldValue T {quadrants['Q1']['T']}
        );
    }}
"""

# Quadrant 2: [0.5,1.0)x[0,0.5)
setFieldsDict_content += f"""
    boxToCell
    {{
        type    boxToCell;
        box     (0.5 0 0) (1.0 0.5 0.5);
        fieldValues
        (
            volVectorFieldValue U ({quadrants['Q2']['vx']} {quadrants['Q2']['vy']} 0)
            volScalarFieldValue p {quadrants['Q2']['p']}
            volScalarFieldValue T {quadrants['Q2']['T']}
        );
    }}
"""

# Quadrant 3: [0,0.5)x[0.5,1.0)
setFieldsDict_content += f"""
    boxToCell
    {{
        type    boxToCell;
        box     (0 0.5 0) (0.5 1.0 0.5);
        fieldValues
        (
            volVectorFieldValue U ({quadrants['Q3']['vx']} {quadrants['Q3']['vy']} 0)
            volScalarFieldValue p {quadrants['Q3']['p']}
            volScalarFieldValue T {quadrants['Q3']['T']}
        );
    }}
"""

# Quadrant 4: [0.5,1.0)x[0.5,1.0)
setFieldsDict_content += f"""
    boxToCell
    {{
        type    boxToCell;
        box     (0.5 0.5 0) (1.0 1.0 0.5);
        fieldValues
        (
            volVectorFieldValue U ({quadrants['Q4']['vx']} {quadrants['Q4']['vy']} 0)
            volScalarFieldValue p {quadrants['Q4']['p']}
            volScalarFieldValue T {quadrants['Q4']['T']}
        );
    }}
);

"""

# Write to system/setFieldsDict
with open("system/setFieldsDict", "w") as f:
    f.write(setFieldsDict_content)

print("Random initial conditions generated and setFieldsDict created.")
for q, vals in quadrants.items():
    print(f"{q}: rho={vals['rho']}, p={vals['p']}, U=({vals['vx']}, {vals['vy']}) -> T={vals['T']}")
