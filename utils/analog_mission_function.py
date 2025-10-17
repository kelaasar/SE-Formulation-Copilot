import pandas as pd
from pathlib import Path  # Import Path for robust path handling



class Tools:
    def __init__(self):
        # Construct the path to your Excel file relative to the script
        self.excel_file_path = "utils/Mission CML1 Database.xlsx"
        self.df = pd.read_excel(self.excel_file_path)

    def filter_past_missions(
        self, TargetBody=None, Element=None, Category=None, params=None
    ):
        """
        Returns past missions and mission proposals based on the Target Body, Element, and mission Category. The target body should
        be selected as the closest appropriate body. For instance, a mission to Diemos should be searched as "Mars." For params, the complete
        name of the column as listed in this docstring must be passed. If the function returns '(blank)' or None, that means that the information is missing from the mission database.

        Inputs:
            TargetBody : Str
                Permitted Values:
                    Earth
                    Moon
                    Mars
                    Comet/Asteroid
                    Inner Planets
                    Outer Planets and their Moons
                    None (returns results for all missions targets)

            Element : Str
                Permitted Values:
                    Orbiter/Flyby
                    Observatory
                    Carrier
                    Descent or Entry Vehicle
                    Rover
                    Lander
                    Ascent Vehicle
                    Earth Return Vehicle
                    Probe, Subsat, Goldola, Balloon
                    None (returns results for all mission targets)

            Category : Str
                Permitted Values:
                    Flagship
                    Large
                    Medium
                    Small
                    None
        """
        if Category == "None":
            Category = None
        if TargetBody == "None":
            TargetBody = None
        if Element == "None":
            Element = None
        if self.df is None:
            return "Error: Mission data is not available. Please check the Excel file path and permissions."

        df = self.df.copy()  # Work on a copy to avoid modifying the original

        final_filtered_df = filter(df, TargetBody, Element, Category)
        return final_filtered_df


def filter(
    df,
    TargetBody=None,
    Element=None,
    Category=None,
):
    vehicleType = {
        "Orbiter": [
            "Orbiter",
            "orbiter, SRM stage",
            "two orbiters",
        ],
        "Orbiter/Flyby": [
            "Orbiter",
            "Flyby",
            "Spacecraft",
            "Sample return touch and go",
            "flyby with impactors",
            "Orbiter (Occulter)",
            "sample return",
            "flyby s/c, impactor",
            "orbiter, sample return capsule",
            "two orbiters",
            "orbiter, SRM stage",
        ],
        "Observatory": ["Observatory"],
        "Carrier": ["Carrier"],
        "Descent or Entry Vehicle": [
            "EDL",
            "Entry vehicle",
            "Descent Vehicle",
            "Descent Vehicle (MSL Skycrane)",
            "Entry",
        ],
        "Rover": ["Rover"],
        "Lander": [
            "Lander",
            "Orbiter and Lander",
            "cruise stage, lander, backshell, heatshield",
        ],
        "Ascent Vehicle": ["Ascent Vehicle", "Lunar ascent module"],
        "Earth Return Vehicle": ["Earth Return Vehicle"],
        "Probe, Subsat, Goldola, Ballloon": [
            "Balloon",
            "Probe",
            "Gondola",
            "Gondola/ Balloon",
            "Subsat",
        ],
    }
    targetBodies = {
        "Earth": [
            "Earth",
            "L1",
            "L2",
            "Heliocentric Earth trailing orbit",
            "L3",
            "Near Earth Objects",
            "Extra solar planets",
            "astrophyics",
            "Sun",
        ],
        "Moon": ["Moon"],
        "Mars": ["Mars", "Diemos"],
        "Comet/Asteroid": [
            "Asteroid",
            "Comet",
            "Near Earth Objects",
            "Trojan Asteroid",
            "Comet Wild2",
            "Apophis",
            "Comet 46P/Wirtanen",
            "Didymos",
        ],
        "Inner Planets": ["Venus", "Mercury"],
        "Outer Planets and their Moons": [
            "Jupiter",
            "Europa",
            "Uranus",
            "Titan",
            "Ganymede",
            "Saturn",
            "Io",
            "Enceladus and Titan",
            "Titan",
            "Enceladus",
            "Jupiter/Io",
        ],
    }

    targetBodyMask = [True] * df.shape[0]
    if TargetBody is not None:
        targetBodyMask = [False] * df.shape[0]

        if isinstance(TargetBody, list):
            for body in TargetBody:
                body = targetBodies[body]
                for ind in body:
                    targetBodyMask = [
                        a or b for a, b in zip(targetBodyMask, df["Target Body"] == ind)
                    ]
        else:
            body = targetBodies[TargetBody]
            for ind in body:
                targetBodyMask = [
                    a or b for a, b in zip(targetBodyMask, df["Target Body"] == ind)
                ]
    elementMask = [True] * df.shape[0]
    if Element is not None:
        elementMask = [False] * df.shape[0]

        if isinstance(Element, list):
            for veh in Element:
                veh = vehicleType[veh]
                for ind in veh:
                    elementMask = [
                        a or b for a, b in zip(elementMask, df["Element"] == ind)
                    ]
        else:
            Element = vehicleType[Element]
            for ind in Element:
                elementMask = [a or b for a, b in zip(elementMask, df["Element"] == ind)]
    categoryMask = [True] * df.shape[0]
    if Category is not None:
        categoryMask = [False] * df.shape[0]
        if isinstance(Category, list):
            for cat in Category:
                categoryMask = [
                    a or b for a, b in zip(categoryMask, df["Category"] == cat)
                ]
        else:
            categoryMask = [
                a or b for a, b in zip(categoryMask, df["Category"] == Category)
            ]

    mask = [a and b and c for a, b, c in zip(targetBodyMask, elementMask, categoryMask)]
    return df[mask]
