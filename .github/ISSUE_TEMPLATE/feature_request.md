name: Feature Request
description: Suggest an idea or extension for this project
labels: ["enhancement"]
body:
  - type: textarea
    id: feature
    attributes:
      label: Proposal
      description: Describe the proposed optimizer variant, schedule, or experiment.
    validations:
      required: true
  - type: textarea
    id: rationale
    attributes:
      label: Scientific Rationale
      description: Why is this feature needed and how does it uphold tuning parity?
