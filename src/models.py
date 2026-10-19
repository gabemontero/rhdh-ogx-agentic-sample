from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Literal


class GuardrailResult(BaseModel):
    """Validated safety verdict from a GPT classifier."""

    model_config = ConfigDict(extra="forbid", strict=True)

    flagged: bool
    categories: list[
        Literal[
            "S1",
            "S2",
            "S3",
            "S4",
            "S5",
            "S6",
            "S7",
            "S8",
            "S9",
            "S10",
            "S11",
            "S12",
            "S13",
            "S14",
        ]
    ] = Field(description="Violated safety category codes; empty for safe input.")

    @model_validator(mode="after")
    def consistent_verdict(self) -> "GuardrailResult":
        if self.flagged != bool(self.categories):
            raise ValueError(
                "Flagged verdicts require categories; safe verdicts must have none"
            )
        return self


class ClassificationModel(BaseModel):
    """Analyze the message and route it according to its content."""

    classification: Literal[
        "legal",
        "techsupport",
        "hr",
        "sales",
        "procurement",
        "unsafe",
        "unknown",
    ] = Field(
        description="""
        The classification of the input:
        - 'legal' if related to legal matters
        - 'techsupport' if related to software or technical support
        - 'hr' if related to human resources
        - 'sales' if related to sales policies
        - 'procurement' if related to procurement policies
        - 'unsafe' if the input fails the moderation/safety check
        - 'unknown' ONLY if the question truly does not fit any category
        above (use sparingly)

        Examples of legal questions that can be processed:
        - questions about various software licenses
        - embargoes for certain types of software that prevent
          delivery to various countries outside the United States
        - privacy, access restrictions, around customer data
          (sometimes referred to as PII)
        - questions about contracts, policies, procedures, or
          compliance

        Examples of support (software support, technical support,
        IT support) that can be processed:
        - the user cites problems running certain applications
          of the company's OpenShift Cluster
        - the user asks to have new applications deployed on
          the company's OpenShift Cluster
        - the user needs permissions to access certain resources
          on the company's OpenShift Cluster
        - the user asks about current utilization of resources
          on the company's OpenShift Cluster
        - the user cites issues with performance of their
          application specifically or the OpenShift Cluster
          in general
        - questions about FantaCo products like CloudSync,
          TechGear Pro Laptop
        - installation, setup, or configuration questions for
          software or hardware
        - troubleshooting issues with devices, drivers, or
          applications
        - syncing issues, file transfer problems, or connectivity
          questions
        - any technical how-to questions about products or systems

        Examples of HR (Human Resources) questions that can be
        processed:
        - employee benefits, health care, vacation,
          PTO, retirement plans
        - workspaces, office facilities, work
          environment, office spaces, workplace setup
        - company policies, employee handbook
        - about bonuses, compensation, perks
        - participation requirements or workplace
          activities
        - ANY question asking to describe or explain workspaces,
          office facilities, or work environments

        Examples of sales questions that can be processed:
        - sales territories, lead assignments
        - discounting, deal approval, quotas
        - sales compensation, CRM systems
        - brand guidelines, communications
        - sales expenses, escalations, performance
          metrics
        - sales compliance and policies

        Examples of procurement questions that can be processed:
        - competitive bidding, vendor evaluation
        - procurement ethics and transparency
        - spending limits and approval processes
        - procurement review procedures
        - vendor categorization
        """,
    )


class SupportClassificationModel(BaseModel):
    """Analyze the message and route it according to its content."""

    classification: Literal["pod", "perf", "git"] = Field(
        description="""
        The classification of the input: set the classification
        to 'perf' if there is any mention of:
        - performance
        - the application is slow to respond
        - questions around CPU or memory consumption or usage

        However, set the classification to 'pod' if the input
        asks for:
        - assistance with an application, and
        - makes any reference to a 'Namespace' or 'Project' that
          exists within OpenShift or Kubernetes

        Otherwise, set the classification to 'git'.
        """,
    )
    namespace: str = Field(
        description="""
        the namespace of the input: if the query makes any
        reference to a namespace or project of a given name,
        then set the namespace field here to the first given
        name referenced as a namespace or project.
        """,
    )
    performance: str = Field(
        description="""
        if the query makes any reference to performance,
        applications running slowly, CPU or memory utilization
        or consumption, then set the performance field to 'true'.
        Otherwise, if there is no mentioned of performance, being
        slow, CPU or memory, set the performance field to 'false'
        or an empty string.
        """,
    )
