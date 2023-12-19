import copy
from documents_toolkit.document_structure import DocumentStructure


def get_problem_set(number_of_problems: int) -> DocumentStructure:
    example_set = [copy.deepcopy(EXERCISE) for i in range(number_of_problems)]
    for i, example in enumerate(example_set):
        example.name = f"Problem {i + 1}"

    return DocumentStructure(
        name="Problem Set",
        description="Set of problems, each with a step-by-step solution, each of varying difficulty.",
        structure=example_set
    )


EXERCISE = DocumentStructure(
    name="Exercise",
    description="Problem with a step-by-step solution.",
    structure=[
        DocumentStructure(
            name="Question",
            description="What is given, relevant definitions for the problem, what needs to be proven, solved, etc."
                        "Written as rigorously and didactic as possible"
        ),
        DocumentStructure(
            name="Solution",
            description="Step-by-step solution to the given problem. Written as rigorously and didactic as possible"
        )
    ]
)

STRUCTURES = [
    DocumentStructure(
        name="Math and Physics Undergraduate Lecture",
        description="Template structure for lecture documents for undergraduate math and physics courses. "
                    "It focuses on presenting complex scientific concepts in an organized and engaging manner, "
                    "integrating theoretical explanations with practical applications and problem-solving.",
        structure=[
            DocumentStructure(
                name="Learning Objectives",
                description="Specific objectives outlining the mathematical or physical concepts students "
                            "will learn."
            ),
            DocumentStructure(
                name="Theoretical Background",
                description="Background information and fundamental theories relevant to the topic, "
                            "including historical developments in math or physics that led to these theories."
            ),
            DocumentStructure(
                name="Core Concepts and Formulas",
                description="Detailed explanation of core concepts, key formulas, and their derivations. "
                            "Includes theorems, equations, and principles specific to the topic."
            ),
            get_problem_set(number_of_problems=3),
            DocumentStructure(
                name="Summary and Recap",
                description="Concluding the lecture with a summary of key points and concepts covered, "
                            "reinforcing learning and highlighting connections between topics."
            ),
            DocumentStructure(
                name="Homework",
                description="Assignments or problem sets for students to work on after the lecture."
            )
        ]
    ),

    DocumentStructure(
        name="Math and Physics Undergraduate Tutorial",
        description="Template structure for tutorials in undergraduate math and physics courses. "
                    "It encompasses key aspects of a tutorial, from introductory content to theoretical "
                    "foundations and worked examples ensuring a well-rounded educational experience.",
        structure=[
            DocumentStructure(
                name="Introduction to the Topic",
                description="Presents basic introduction to key concepts, historical context, and prerequisites."
            ),
            DocumentStructure(
                name="Theoretical Foundations",
                description="Detailed explanation of main theories, important formulas and their derivations, "
                            "accompanied by visual aids."
            ),
            get_problem_set(number_of_problems=3),
            DocumentStructure(
                name="Summary and Key Takeaways",
                description="Recap of main points covered, key takeaways and formulas, and suggestions for further "
                            "reading."
            )
        ]
    ),

    DocumentStructure(
        name="Math and Physics Problem Set",
        description="Template structure for problem sets in undergraduate math and physics courses."
                    "It outlines the objective of the problem set and includes problem-solving tips along "
                    "the problems which vary in complexity and scope.",
        structure=[
            DocumentStructure(
                name="Problem Set Introduction",
                description="An introductory section that outlines the objectives of the problem set, "
                            "the topics covered, and any necessary instructions or guidelines."
            ),
            get_problem_set(number_of_problems=5),
            DocumentStructure(
                name="Problem-Solving Tips",
                description="Helpful hints or strategies for approaching and solving the problems, "
                            "aimed at guiding students without giving away the solutions."
            ),
            DocumentStructure(
                name="Submission Guidelines",
                description="Clear instructions on how to complete and submit the problem set, "
                            "including any formatting requirements and deadlines."
            )
        ]
    )
]

STRUCTURES_MAP = {struct.name: struct for struct in STRUCTURES}
