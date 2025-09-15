"""
Governance System - Constitutional Governance and Stakeholder Management
=======================================================================

The GovernanceSystem provides comprehensive governance mechanisms for the
constitutional framework, including stakeholder management, decision-making
processes, consensus building, and democratic oversight of system operations.

Key Features:
- Multi-stakeholder governance with role-based permissions
- Democratic decision-making processes and voting mechanisms
- Consensus building algorithms for policy changes
- Constitutional amendment procedures with proper approvals
- Transparency and accountability mechanisms
- Conflict resolution and dispute management
- Governance audit trails and reporting

Architecture:
The GovernanceSystem operates as a democratic oversight layer ensuring that
all constitutional changes and major system decisions follow proper
governance procedures with appropriate stakeholder involvement and approval.
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
import uuid
import asyncio
from collections import defaultdict


class StakeholderRole(Enum):
    """Stakeholder roles in constitutional governance."""

    ADMINISTRATOR = "administrator"  # System administrators
    SUPERVISOR = "supervisor"  # Human supervisors
    USER = "user"  # End users
    DEVELOPER = "developer"  # System developers
    AUDITOR = "auditor"  # External auditors
    SYSTEM = "system"  # Automated system processes


class VoteType(Enum):
    """Types of governance votes."""

    SIMPLE_MAJORITY = "simple_majority"  # >50% approval
    SUPERMAJORITY = "supermajority"  # >66% approval
    UNANIMOUS = "unanimous"  # 100% approval
    WEIGHTED = "weighted"  # Weighted by role importance


class ProposalStatus(Enum):
    """Status of governance proposals."""

    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    VOTING = "voting"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"
    WITHDRAWN = "withdrawn"


class GovernanceAction(Enum):
    """Types of governance actions."""

    POLICY_CREATE = "policy_create"
    POLICY_MODIFY = "policy_modify"
    POLICY_DELETE = "policy_delete"
    CONSTITUTIONAL_AMENDMENT = "constitutional_amendment"
    EMERGENCY_OVERRIDE = "emergency_override"
    STAKEHOLDER_ADD = "stakeholder_add"
    STAKEHOLDER_REMOVE = "stakeholder_remove"
    SYSTEM_CONFIGURATION = "system_configuration"


@dataclass
class Stakeholder:
    """Stakeholder in the governance system."""

    stakeholder_id: str
    name: str
    role: StakeholderRole
    email: Optional[str] = None
    voting_weight: float = 1.0
    permissions: Set[str] = field(default_factory=set)
    active: bool = True
    joined_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class GovernanceProposal:
    """Governance proposal for system changes."""

    proposal_id: str
    title: str
    description: str
    action_type: GovernanceAction
    proposed_by: str
    content: Dict[str, Any]
    status: ProposalStatus = ProposalStatus.DRAFT
    vote_type: VoteType = VoteType.SIMPLE_MAJORITY
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    voting_deadline: Optional[datetime] = None
    votes: Dict[str, bool] = field(default_factory=dict)  # stakeholder_id -> vote
    comments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class GovernanceSystem:
    """Constitutional governance and stakeholder management system."""

    def __init__(self):
        """Initialize the Governance System."""
        self.stakeholders: Dict[str, Stakeholder] = {}
        self.proposals: Dict[str, GovernanceProposal] = {}
        self.voting_active: Dict[str, bool] = {}  # proposal_id -> active status
        self.governance_config = {
            "minimum_voting_period_hours": 24,
            "maximum_voting_period_hours": 168,  # 1 week
            "quorum_threshold": 0.5,  # 50% of eligible stakeholders must vote
            "supermajority_threshold": 0.66,
            "emergency_override_enabled": True,
            "anonymous_voting": False,
        }

        self.governance_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("GovernanceSystem")

        # Initialize system administrator
        self._create_system_admin()

    def _create_system_admin(self) -> None:
        """Create default system administrator."""
        admin_id = "SYSTEM_ADMIN_001"
        if admin_id not in self.stakeholders:
            admin = Stakeholder(
                stakeholder_id=admin_id,
                name="NFCS System Administrator",
                role=StakeholderRole.ADMINISTRATOR,
                voting_weight=2.0,  # Higher weight for system admin
                permissions={
                    "create_proposal",
                    "vote_on_proposal",
                    "manage_stakeholders",
                    "emergency_override",
                    "system_configuration",
                    "view_all_data",
                },
            )
            self.stakeholders[admin_id] = admin
            self.logger.info("Created system administrator")

    def add_stakeholder(
        self, stakeholder_id: str, name: str, role: StakeholderRole, **kwargs
    ) -> Stakeholder:
        """Add a new stakeholder to the governance system."""
        if stakeholder_id in self.stakeholders:
            raise ValueError(f"Stakeholder {stakeholder_id} already exists")

        # Set default permissions based on role
        default_permissions = self._get_default_permissions(role)
        permissions = kwargs.get("permissions", default_permissions)

        stakeholder = Stakeholder(
            stakeholder_id=stakeholder_id,
            name=name,
            role=role,
            permissions=set(permissions),
            **{k: v for k, v in kwargs.items() if k != "permissions"},
        )

        self.stakeholders[stakeholder_id] = stakeholder
        self.logger.info(f"Added stakeholder: {name} ({role.value})")
        return stakeholder

    def _get_default_permissions(self, role: StakeholderRole) -> Set[str]:
        """Get default permissions for a stakeholder role."""
        permission_map = {
            StakeholderRole.ADMINISTRATOR: {
                "create_proposal",
                "vote_on_proposal",
                "manage_stakeholders",
                "emergency_override",
                "system_configuration",
                "view_all_data",
            },
            StakeholderRole.SUPERVISOR: {
                "create_proposal",
                "vote_on_proposal",
                "view_proposals",
                "emergency_override",
                "view_audit_logs",
            },
            StakeholderRole.DEVELOPER: {
                "create_proposal",
                "vote_on_proposal",
                "view_proposals",
                "system_configuration",
            },
            StakeholderRole.AUDITOR: {
                "vote_on_proposal",
                "view_proposals",
                "view_all_data",
                "view_audit_logs",
            },
            StakeholderRole.USER: {"vote_on_proposal", "view_proposals"},
            StakeholderRole.SYSTEM: {"view_proposals", "view_audit_logs"},
        }
        return permission_map.get(role, {"view_proposals"})

    def create_proposal(
        self,
        title: str,
        description: str,
        action_type: GovernanceAction,
        proposed_by: str,
        content: Dict[str, Any],
        **kwargs,
    ) -> GovernanceProposal:
        """Create a new governance proposal."""
        # Verify proposer has permission
        proposer = self.stakeholders.get(proposed_by)
        if not proposer or "create_proposal" not in proposer.permissions:
            raise PermissionError(f"Stakeholder {proposed_by} cannot create proposals")

        proposal_id = f"PROP_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

        # Set voting deadline
        voting_hours = kwargs.get(
            "voting_period_hours", self.governance_config["minimum_voting_period_hours"]
        )
        voting_deadline = datetime.now(timezone.utc) + datetime.timedelta(hours=voting_hours)

        proposal = GovernanceProposal(
            proposal_id=proposal_id,
            title=title,
            description=description,
            action_type=action_type,
            proposed_by=proposed_by,
            content=content,
            voting_deadline=voting_deadline,
            **{k: v for k, v in kwargs.items() if k != "voting_period_hours"},
        )

        self.proposals[proposal_id] = proposal
        self.logger.info(f"Created proposal: {title} by {proposer.name}")
        return proposal

    def submit_proposal(self, proposal_id: str) -> bool:
        """Submit proposal for review and voting."""
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            return False

        proposal.status = ProposalStatus.SUBMITTED
        self.logger.info(f"Submitted proposal: {proposal.title}")
        return True

    def start_voting(self, proposal_id: str) -> bool:
        """Start voting on a proposal."""
        proposal = self.proposals.get(proposal_id)
        if not proposal or proposal.status != ProposalStatus.SUBMITTED:
            return False

        proposal.status = ProposalStatus.VOTING
        self.voting_active[proposal_id] = True
        self.logger.info(f"Started voting on proposal: {proposal.title}")
        return True

    def cast_vote(self, proposal_id: str, stakeholder_id: str, vote: bool) -> bool:
        """Cast a vote on a proposal."""
        proposal = self.proposals.get(proposal_id)
        stakeholder = self.stakeholders.get(stakeholder_id)

        if not proposal or not stakeholder:
            return False

        if proposal.status != ProposalStatus.VOTING:
            return False

        if "vote_on_proposal" not in stakeholder.permissions:
            return False

        # Check voting deadline
        if proposal.voting_deadline and datetime.now(timezone.utc) > proposal.voting_deadline:
            return False

        proposal.votes[stakeholder_id] = vote
        stakeholder.last_activity = datetime.now(timezone.utc)

        self.logger.info(f"Vote cast by {stakeholder.name}: {'YES' if vote else 'NO'}")
        return True

    def finalize_vote(self, proposal_id: str) -> Dict[str, Any]:
        """Finalize voting and determine outcome."""
        proposal = self.proposals.get(proposal_id)
        if not proposal or proposal.status != ProposalStatus.VOTING:
            return {"success": False, "reason": "Invalid proposal or status"}

        # Calculate vote results
        total_votes = len(proposal.votes)
        yes_votes = sum(1 for vote in proposal.votes.values() if vote)
        no_votes = total_votes - yes_votes

        # Get eligible voters
        eligible_voters = [
            s
            for s in self.stakeholders.values()
            if s.active and "vote_on_proposal" in s.permissions
        ]
        total_eligible = len(eligible_voters)

        # Check quorum
        quorum_met = total_votes >= (total_eligible * self.governance_config["quorum_threshold"])

        # Determine if proposal passes based on vote type
        passes = self._determine_vote_outcome(proposal, yes_votes, total_votes)

        # Update proposal status
        if quorum_met and passes:
            proposal.status = ProposalStatus.APPROVED
            outcome = "APPROVED"
        else:
            proposal.status = ProposalStatus.REJECTED
            outcome = "REJECTED"

        self.voting_active[proposal_id] = False

        result = {
            "success": True,
            "outcome": outcome,
            "yes_votes": yes_votes,
            "no_votes": no_votes,
            "total_votes": total_votes,
            "quorum_met": quorum_met,
            "quorum_threshold": self.governance_config["quorum_threshold"],
            "total_eligible": total_eligible,
        }

        # Log to governance history
        self.governance_history.append(
            {
                "proposal_id": proposal_id,
                "title": proposal.title,
                "outcome": outcome,
                "finalized_at": datetime.now(timezone.utc).isoformat(),
                "vote_results": result,
            }
        )

        self.logger.info(f"Vote finalized: {proposal.title} - {outcome}")
        return result

    def _determine_vote_outcome(
        self, proposal: GovernanceProposal, yes_votes: int, total_votes: int
    ) -> bool:
        """Determine if proposal passes based on vote type."""
        if total_votes == 0:
            return False

        approval_rate = yes_votes / total_votes

        if proposal.vote_type == VoteType.UNANIMOUS:
            return approval_rate == 1.0
        elif proposal.vote_type == VoteType.SUPERMAJORITY:
            return approval_rate >= self.governance_config["supermajority_threshold"]
        else:  # SIMPLE_MAJORITY or WEIGHTED
            return approval_rate > 0.5

    def get_active_proposals(self) -> List[GovernanceProposal]:
        """Get all active proposals."""
        return [
            p
            for p in self.proposals.values()
            if p.status in [ProposalStatus.SUBMITTED, ProposalStatus.VOTING]
        ]

    def get_proposal_summary(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a proposal including voting status."""
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            return None

        return {
            "proposal_id": proposal.proposal_id,
            "title": proposal.title,
            "status": proposal.status.value,
            "action_type": proposal.action_type.value,
            "proposed_by": proposal.proposed_by,
            "votes_count": len(proposal.votes),
            "voting_deadline": (
                proposal.voting_deadline.isoformat() if proposal.voting_deadline else None
            ),
            "created_at": proposal.created_at.isoformat(),
        }

    def get_governance_status(self) -> Dict[str, Any]:
        """Get overall governance system status."""
        active_proposals = len(self.get_active_proposals())
        total_stakeholders = len([s for s in self.stakeholders.values() if s.active])

        return {
            "total_stakeholders": total_stakeholders,
            "active_proposals": active_proposals,
            "total_proposals": len(self.proposals),
            "governance_config": self.governance_config.copy(),
            "stakeholder_roles": {
                role.value: len(
                    [s for s in self.stakeholders.values() if s.role == role and s.active]
                )
                for role in StakeholderRole
            },
        }
