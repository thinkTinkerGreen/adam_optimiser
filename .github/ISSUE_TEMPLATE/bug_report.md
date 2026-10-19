name: Bug Report
description: Create a report to help us fix a bug or inconsistency
labels: ["bug"]
body:
  - type: markdown
    attributes:
      value: Thanks for reporting a bug. Please ensure you are testing against a clean environment.
  - type: textarea
    id: what-happened
    attributes:
      label: What happened?
      description: Describe what happened and what you expected to happen instead.
    validations:
      required: true
  - type: textarea
    id: reproduction
    attributes:
      label: Steps to Reproduce
      description: Provide clear reproduction steps (including random seed and hardware specs).
    validations:
      required: true
  - type: textarea
    id: logs
    attributes:
      label: Relevant logs or error output
      render: shell
