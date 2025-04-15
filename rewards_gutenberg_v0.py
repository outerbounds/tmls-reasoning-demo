
import re
import html
from xml.etree import ElementTree as ET
from typing import Tuple, List, Dict, Any, Optional

import torch
from torchtune.modules.transforms.tokenizers import ModelTokenizer

# TODO: Scoop this from download_src_data module to ensure consistency.
VALID_ERAS = [
    "renaissance",
    "enlightenment",
    "victorian",
    "edwardian",
    "modern"
]

class RewardServer(object):

    def __init__(self):
        pass

    def check_outside_text(self, text: str) -> tuple[bool, str]:
        """
        More robust function to detect text outside the required tags.

        Args:
            text: The text to analyze

        Returns:
            tuple: (has_outside_text, outside_text)
        """
        # First strip whitespace
        text = text.strip()

        # Use regex to extract all tag content
        pattern = r"<(think|answer_date|answer_era)>(.*?)</\1>"
        matches = re.findall(pattern, text, re.DOTALL)

        # Create a cleaned version for comparison
        cleaned_text = text

        # Remove all valid tag content
        for tag, content in matches:
            cleaned_text = cleaned_text.replace(f"<{tag}>{content}</{tag}>", "", 1)

        # Strip whitespace again
        cleaned_text = cleaned_text.strip()

        return bool(cleaned_text), cleaned_text


    def extract_tags(self, text: str) -> dict[str, list[str]]:
        """
        Parse XML-like tags from text, with improved handling for malformed XML.

        Args:
            text: Text potentially containing XML tags

        Returns:
            Dictionary with tag content keyed by tag name
        """
        results = {
            "think": [],
            "answer_era": [],
            "answer_date": []
        }

        # First, try regex method (more robust for malformed XML)
        for tag in ["think", "answer_era", "answer_date"]:
            pattern = f"<{tag}>(.*?)</{tag}>"
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                # Trim whitespace from each match
                results[tag] = [match.strip() for match in matches]

        # If regex found matches, return those
        if any(len(v) > 0 for v in results.values()):
            return results

        # Otherwise try the XML parser as fallback
        xml_string = f"<root>{text}</root>"
        try:
            root = ET.fromstring(xml_string)
            for tag in ["think", "answer_era", "answer_date"]:
                results[tag] = [
                    (elem.text.strip() if elem.text else "") 
                    for elem in root.findall(tag)
                ]
            return results
        except ET.ParseError:
            # Return the empty results if both methods fail
            return results


    def shaped_correctness_reward(self, answer: str, completion: str) -> tuple[float, float, Dict[str, Any]]:
        """
        Reward function for gutenberg_eras_tour task with detailed diagnostics.

        Args:
            answer: Ground-truth answer string in format "era (date)"
            completion: Model's unparsed completion string

        Returns:
            tuple: (reward_score, success_flag, details_dict)
        """

        # Rewards can be negative. Max reward is 100.
        # TODO: I've seen some folks speculating rewards in [0, 1] is beneficial.
        reward = 0.0
        success = 0.0

        # Storage for diagnostics.
        details = {
            "ground_truth": {
                "original": answer,
                "era": "",
                "date": ""
            },
            "completion": completion[:100] + ("..." if len(completion) > 100 else ""),  # Truncated for readability
            "extracted_tags": {},
            "format_analysis": {},
            "content_analysis": {},
            "reward_components": [],
            "total_reward": 0.0,
            "success": 0.0
        }

        # Parse the true labels we want the LLM to learn to infer based on its reasoning.
        gt_match = re.match(r'([a-z]+)\s*\((\d+)\)', answer.lower())
        if gt_match:
            gt_era = gt_match.group(1).strip()
            gt_date = gt_match.group(2).strip()
        else:
            # Fallback if parsing fails
            gt_era = answer.lower().strip()
            gt_date = ""
        details["ground_truth"]["era"] = gt_era
        details["ground_truth"]["date"] = gt_date

        # Parse content from the LLM's completion.
        tags = self.extract_tags(completion)
        details["extracted_tags"] = {
            "think": tags["think"],
            "answer_era": tags["answer_era"],
            "answer_date": tags["answer_date"]
        }

        ### FORMATTING PENALTY ### 

        # Max reward: 0.
        # Min reward: -30.
        # Does text exist outside of desired XML format?
        has_outside_text, outside_text = self.check_outside_text(completion)
        details["format_analysis"]["has_outside_text"] = has_outside_text
        if has_outside_text:
            details["format_analysis"]["outside_text"] = outside_text
        # Apply significant penalties for text outside tags
        if has_outside_text:
            # More aggressive penalty for text outside tags
            penalty = min(30.0, len(outside_text) * 0.2)
            reward -= penalty
            details["reward_components"].append({
                "component": "outside_text_penalty",
                "value": -penalty,
                "reason": f"Text found outside required tags: '{outside_text[:30]}...' ({len(outside_text)} chars)"
            })

        ### FORMATTING REWARDS ###

        ## <think> FORMATTING ##
        # Max reward: 10.
        # Min reward: -5.
        if len(tags["think"]) == 1:
            reward += 10.0  # Good reward for having exactly one thinking section
            details["reward_components"].append({
                "component": "think_tag_format",
                "value": 10.0,
                "reason": "Correct: Exactly one <think> tag"
            })
        elif len(tags["think"]) > 1:
            reward += 2.5   # Smaller reward for having thinking, but too many sections
            details["reward_components"].append({
                "component": "think_tag_format",
                "value": 2.5,
                "reason": f"Partial: {len(tags['think'])} <think> tags found (expected 1)"
            })
        else:
            reward -= 5.0   # Penalty for missing thinking section
            details["reward_components"].append({
                "component": "think_tag_format",
                "value": -5.0,
                "reason": "Missing <think> tag"
            })

        ## <answer_era> FORMATTING ##
        # Max reward: 10.
        # Min reward: -5.
        if len(tags["answer_era"]) == 1:
            reward += 10.0  # Increased reward for having exactly one era tag
            details["reward_components"].append({
                "component": "era_tag_format",
                "value": 10.0,
                "reason": "Correct: Exactly one <answer_era> tag"
            })
        elif len(tags["answer_era"]) > 1:
            reward += 2.5   # Some reward for having era tags, but too many
            details["reward_components"].append({
                "component": "era_tag_format",
                "value": 2.5,
                "reason": f"Partial: {len(tags['answer_era'])} <answer_era> tags found (expected 1)"
            })
        else:
            reward -= 10.0  # Stronger penalty for missing era
            details["reward_components"].append({
                "component": "era_tag_format",
                "value": -10.0,
                "reason": "Missing <answer_era> tag"
            })

        ## <answer_date> FORMATTING ##
        # Max reward: 10.
        # Min reward: -5.
        if len(tags["answer_date"]) == 1:
            reward += 10.0  # Increased reward for having exactly one date tag
            details["reward_components"].append({
                "component": "date_tag_format",
                "value": 10.0,
                "reason": "Correct: Exactly one <answer_date> tag"
            })
        elif len(tags["answer_date"]) > 1:
            reward += 2.5   # Some reward for having date tags, but too many
            details["reward_components"].append({
                "component": "date_tag_format",
                "value": 2.5,
                "reason": f"Partial: {len(tags['answer_date'])} <answer_date> tags found (expected 1)"
            })
        else:
            reward -= 10.0  # Stronger penalty for missing date
            details["reward_components"].append({
                "component": "date_tag_format",
                "value": -10.0,
                "reason": "Missing <answer_date> tag"
            })

        ## <answer_era> validation rewards ##
        # Max reward: 5.
        # Min reward: -5.
        if tags["answer_era"]:
            details["content_analysis"]["era"] = {
                "provided": [era.lower() for era in tags["answer_era"]],
                "valid_eras": VALID_ERAS,
                "ground_truth": gt_era
            }

            # Is LLM-provided era in the valid list?
            valid_provided = [era.lower() for era in tags["answer_era"] if era.lower() in VALID_ERAS]
            if valid_provided:
                reward += 5.0  # Bonus for using a valid era from the list
                details["reward_components"].append({
                    "component": "era_validation",
                    "value": 5.0,
                    "reason": f"Used valid era(s): {', '.join(valid_provided)}"
                })
            else:
                reward -= 5.0  # Penalty for using invalid era
                details["reward_components"].append({
                    "component": "era_validation",
                    "value": -5.0,
                    "reason": f"Invalid era(s): {', '.join([era.lower() for era in tags['answer_era']])}"
                })

        ## <answer_era> correctness rewards ##
        # Max reward: 30.
        # Min reward: 0.
        if tags["answer_era"]:
            exact_match = any(gt_era == attempt.lower().strip() for attempt in tags["answer_era"])
            partial_match = any(gt_era in attempt.lower().strip() for attempt in tags["answer_era"])

            details["content_analysis"]["era_match"] = {
                "exact_match": exact_match,
                "partial_match": partial_match
            }

            if exact_match:
                # One of the answer_era tags has the exact right era
                reward += 30.0  # Increased reward for correct era
                details["reward_components"].append({
                    "component": "era_correctness",
                    "value": 30.0,
                    "reason": f"Correct era: {gt_era}"
                })
            elif partial_match:
                # One of the answer_era tags contains the right era as a substring
                reward += 10.0  # Partial reward
                details["reward_components"].append({
                    "component": "era_correctness",
                    "value": 10.0,
                    "reason": f"Partial era match: Contains '{gt_era}'"
                })
            else:
                details["reward_components"].append({
                    "component": "era_correctness",
                    "value": 0.0,
                    "reason": f"Incorrect era: Expected '{gt_era}'"
                })

        ## <answer_date> correctness rewards ##
        # Max reward: 30.
        # Min reward: -5.
        if gt_date and tags["answer_date"]:
            try:
                gt_year = int(gt_date)

                # Parse all dates and track errors
                date_attempts = []
                valid_dates = []

                for attempt in tags["answer_date"]:
                    attempt = attempt.strip()
                    date_attempts.append(attempt)
                    if attempt.isdigit():
                        valid_dates.append(int(attempt))

                details["content_analysis"]["date"] = {
                    "provided": date_attempts,
                    "valid_dates": valid_dates,
                    "ground_truth": gt_year
                }

                if valid_dates:
                    # Find best date attempt (closest to ground truth)
                    best_diff = min(abs(date - gt_year) for date in valid_dates)
                    best_date = next(date for date in valid_dates if abs(date - gt_year) == best_diff)

                    details["content_analysis"]["date"]["best_match"] = {
                        "value": best_date,
                        "difference": best_diff
                    }

                    # Award based on closest date
                    if best_diff == 0:
                        # Exact date match
                        reward += 30.0  # Increased reward
                        details["reward_components"].append({
                            "component": "date_correctness",
                            "value": 30.0,
                            "reason": f"Exact date match: {best_date}"
                        })
                    elif best_diff <= 20:
                        # Within 20 years
                        reward += 20.0  # Increased reward
                        details["reward_components"].append({
                            "component": "date_correctness",
                            "value": 20.0,
                            "reason": f"Close date match: {best_date} (within 20 years of {gt_year})"
                        })
                    elif best_diff <= 50:
                        # Within 50 years
                        reward += 10.0
                        details["reward_components"].append({
                            "component": "date_correctness",
                            "value": 10.0,
                            "reason": f"Approximate date: {best_date} (within 50 years of {gt_year})"
                        })
                    elif best_diff <= 100:
                        # Within 100 years
                        reward += 5.0
                        details["reward_components"].append({
                            "component": "date_correctness",
                            "value": 5.0,
                            "reason": f"Distant date: {best_date} (within 100 years of {gt_year})"
                        })
                    else:
                        # More than 100 years off
                        reward -= 5.0  # Small penalty for very wrong date
                        details["reward_components"].append({
                            "component": "date_correctness",
                            "value": -5.0,
                            "reason": f"Incorrect date: {best_date} (more than 100 years from {gt_year})"
                        })
                else:
                    # No valid numeric dates found
                    reward -= 5.0  # Penalty for non-numeric date
                    details["reward_components"].append({
                        "component": "date_correctness",
                        "value": -5.0,
                        "reason": f"Non-numeric date(s): {', '.join(date_attempts)}"
                    })

            except ValueError as e:
                # Penalty for non-numeric date
                reward -= 5.0
                details["reward_components"].append({
                    "component": "date_correctness",
                    "value": -5.0,
                    "reason": f"Date parsing error: {str(e)}"
                })

        ## Success criteria ##
        # Both era and date must be correct AND format must be perfect
        perfect_format = (
            len(tags["think"]) == 1 and 
            len(tags["answer_era"]) == 1 and 
            len(tags["answer_date"]) == 1 and
            not has_outside_text
        )

        correct_era = (
            tags["answer_era"] and 
            tags["answer_era"][0].lower().strip() == gt_era
        )

        correct_date = (
            gt_date and
            tags["answer_date"] and
            tags["answer_date"][0].isdigit() and 
            abs(int(tags["answer_date"][0]) - int(gt_date)) <= 20
        )

        details["success_criteria"] = {
            "perfect_format": perfect_format,
            "correct_era": correct_era,
            "correct_date": correct_date
        }

        if perfect_format and correct_era and correct_date:
            reward = 100.0
            success = 1.0
            details["reward_components"].append({
                "component": "perfect_answer",
                "value": "100.0 (overwrites previous)",
                "reason": "Perfect format and correct answers"
            })

        # Store final reward and success in details
        details["total_reward"] = reward
        details["success"] = success

        return reward, success, details


    def batch_shaped_correctness_reward(
        self,
        tokenizer: ModelTokenizer, 
        completions: torch.Tensor, 
        answers: List[str],
        details_report: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[Dict[str, Any]]]]:
        """
        Apply the shaped reward function to a batch of completions.
        DO NOT change the signature of this function, or it will break the design pattern of how this object method is invoked in grpo_full_finetune_distributed.py.

        Args:
            tokenizer: Tokenizer for decoding completions
            completions: Tensor of token IDs
            answers: List of ground truth answers
            details_report: Whether to generate detailed diagnostic reports

        Returns:
            Tuple of (rewards, successes, optional details list)
        """
        batch_size, grpo_size, *_ = completions.shape
        rewards = torch.zeros(batch_size, grpo_size, dtype=torch.float32)
        successes = torch.zeros(batch_size, grpo_size, dtype=torch.float32)

        # Create container for details if requested
        details_list = [] if details_report else None

        # Process each completion in the batch
        for b in range(batch_size):
            batch_details = [] if details_report else None

            for g in range(grpo_size):
                # Decode the completion
                text_completion = tokenizer.decode(
                    completions[b, g].tolist()
                )

                # Calculate reward, success, and details
                reward, success, details = self.shaped_correctness_reward(
                    answer=answers[b], 
                    completion=text_completion
                )

                # Store results
                rewards[b, g] = reward
                successes[b, g] = success

                # Store details if requested
                if details_report:
                    # Add batch and group indices
                    details["batch_idx"] = b
                    details["group_idx"] = g
                    batch_details.append(details)

            # Add batch details to the main list
            if details_report:
                details_list.append(batch_details)

        return rewards, successes, details_list


    # Helper function to print a readable summary of the details
    def print_reward_details_summary(self, details: Dict[str, Any]) -> None:
        """
        Print a human-readable summary of the reward details.

        Args:
            details: The details dictionary from shaped_correctness_reward
        """
        print(f"=== Reward Calculation Summary ===")
        print(f"Ground Truth: Era='{details['ground_truth']['era']}', Date='{details['ground_truth']['date']}'")
        print(f"Completion: {details['completion']}")
        print("\nExtracted Tags:")
        print(f"  Think: {len(details['extracted_tags']['think'])} tag(s)")
        print(f"  Era: {len(details['extracted_tags']['answer_era'])} tag(s)")
        print(f"  Date: {len(details['extracted_tags']['answer_date'])} tag(s)")

        print("\nFormat Analysis:")
        if details['format_analysis'].get('has_outside_text', False):
            print(f"  ❌ Text outside tags: {details['format_analysis']['outside_text']}")
        else:
            print(f"  ✓ No text outside tags")

        print("\nContent Analysis:")
        if 'era' in details['content_analysis']:
            print(f"  Era provided: {details['content_analysis']['era']['provided']}")
            match_status = "❌ No match"
            if details['content_analysis'].get('era_match', {}).get('exact_match', False):
                match_status = "✓ Exact match"
            elif details['content_analysis'].get('era_match', {}).get('partial_match', False):
                match_status = "~ Partial match"
            print(f"  Era match: {match_status}")

        if 'date' in details['content_analysis']:
            print(f"  Date provided: {details['content_analysis']['date']['provided']}")
            if 'best_match' in details['content_analysis']['date']:
                best = details['content_analysis']['date']['best_match']
                print(f"  Best date: {best['value']} (diff: {best['difference']} years)")

        print("\nReward Components:")
        for component in details['reward_components']:
            print(f"  {component['component']}: {component['value']} - {component['reason']}")

        print(f"\nTotal Reward: {details['total_reward']}")
        print(f"Success: {details['success']}")

        if 'success_criteria' in details:
            criteria = details['success_criteria']
            print("\nSuccess Criteria:")
            print(f"  Format perfect: {'✓' if criteria['perfect_format'] else '❌'}")
            print(f"  Era correct: {'✓' if criteria['correct_era'] else '❌'}")
            print(f"  Date correct: {'✓' if criteria['correct_date'] else '❌'}")

        print("================================")

    def display_responses(
        self,
        responses: torch.Tensor,
        tokenizer: ModelTokenizer,
        grpo_size: int,
        advantages: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        successes: Optional[torch.Tensor] = None,
        details: Optional[List[List[Dict[str, Any]]]] = None,
        show_n: Optional[int] = None
    ) -> str:
        """
        Display responses with rewards, advantages, and detailed diagnostics in a visually appealing format.

        Args:
            responses: Tensor of token IDs
            tokenizer: Tokenizer for decoding responses
            grpo_size: Size of the policy optimization group
            advantages: Optional tensor of advantages
            rewards: Optional tensor of rewards
            successes: Optional tensor of successes
            details: Optional list of reward calculation details
            show_n: Optional maximum number of responses to display

        Returns:
            HTML string for displaying the responses
        """
        batch_size = responses.shape[0]

        # Helper function to safely get values from tensors with different shapes
        def get_item_value(tensor, batch_idx, group_idx):
            if tensor is None:
                return None

            if tensor.dim() == 1:
                # Handle 1D tensor [grpo_size]
                return tensor[group_idx].item()
            else:
                # Handle 2D tensor [batch_size, grpo_size]
                return tensor[batch_idx][group_idx].item()

        html_output = """
        <style>
            .response-container {
                margin: 20px 0;
                border: 1px solid #C4C7AC;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                font-family: 'Courier New', monospace;
                max-width: 100%;
            }
            .response-header {
                background-color: #F0EBE5;
                padding: 10px 15px;
                font-size: 16px;
                font-weight: bold;
                border-bottom: 1px solid #C4C7AC;
                color: #4A4A67;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .response-body {
                background-color: #ffffff;
                color: #4A4A67;
                padding: 15px;
                white-space: pre-wrap;
                word-wrap: break-word;
                line-height: 1.6;
                font-size: 14px;
            }
            .think-tag {
                color: #BE6A1A;
                font-weight: bold;
            }
            .answer-tag {
                color: #2C6846;
                font-weight: bold;
            }
            .answer-era-tag {
                color: #2C6846;
                font-weight: bold;
            }
            .answer-date-tag {
                color: #2C6846;
                font-weight: bold;
            }
            .metrics-container {
                background-color: #F0EBE5;
                border-top: 1px solid #C4C7AC;
                padding: 10px 15px;
            }
            .metric-label {
                color: #4A4A67;
            }
            .metric-score {
                font-family: monospace;
                font-weight: bold;
                padding: 2px 8px;
                border-radius: 4px;
                display: inline-block;
                margin-right: 8px;
            }
            .score-high {
                background-color: #D3EFE0;
                color: #177350;
            }
            .score-medium {
                background-color: #FCF1D6;
                color: #BE6A1A;
            }
            .score-low {
                background-color: #FAD9D8;
                color: #C5393A;
            }
            .success-badge {
                background-color: #177350;
                color: white;
                padding: 3px 8px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }
            .failure-badge {
                background-color: #C5393A;
                color: white;
                padding: 3px 8px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }
            .metrics-toggle {
                cursor: pointer;
                color: #3F7DC9;
                text-decoration: underline;
                font-size: 12px;
                margin-top: 5px;
                display: inline-block;
                font-weight: bold;
            }
            .details-container {
                display: none;
                margin-top: 10px;
                border-top: 1px solid #C4C7AC;
                padding-top: 10px;
            }

            /* Reward details styling */
            .reward-component {
                margin-bottom: 10px;
                padding: 8px;
                border-radius: 4px;
                background-color: #f8f9fa;
            }
            .component-name {
                font-weight: bold;
                color: #4A4A67;
            }
            .component-value {
                font-family: monospace;
                padding: 2px 4px;
                border-radius: 3px;
            }
            .component-value-positive {
                background-color: #D3EFE0;
                color: #177350;
            }
            .component-value-negative {
                background-color: #FAD9D8;
                color: #C5393A;
            }
            .component-reason {
                font-size: 0.9em;
                color: #555;
                margin-top: 4px;
            }
            .check-success {
                color: #177350;
                font-weight: bold;
            }
            .check-fail {
                color: #C5393A;
                font-weight: bold;
            }
            .batch-header {
                margin: 30px 0 10px 0;
                padding: 5px 10px;
                background-color: #E5E7D9;
                border-left: 4px solid #4A4A67;
                color: #4A4A67;
                font-size: 18px;
                font-weight: bold;
            }
        </style>

        <script>
        function toggleDetails(batchIdx, groupIdx) {
            var detailsId = 'details-' + batchIdx + '-' + groupIdx;
            var details = document.getElementById(detailsId);
            var buttonId = 'toggle-' + batchIdx + '-' + groupIdx;
            var toggleBtn = document.getElementById(buttonId);

            if (details) {
                if (details.style.display === 'none' || details.style.display === '') {
                    details.style.display = 'block';
                    if (toggleBtn) toggleBtn.innerText = 'Hide Details';
                } else {
                    details.style.display = 'none';
                    if (toggleBtn) toggleBtn.innerText = 'Show Details';
                }
            }
        }
        </script>
        """

        if show_n is not None:
            grpo_size = min(grpo_size, show_n)

        for b in range(batch_size):
            html_output += f'<div class="batch-header">Batch #{b+1}</div>'

            for g in range(grpo_size):
                # Decode the response
                response_text = tokenizer.decode(responses[b, g].tolist())

                # Determine if this response succeeded
                success_value = get_item_value(successes, b, g) if successes is not None else None
                is_successful = success_value == 1.0 if success_value is not None else None

                # Get reward and advantage if available
                reward_value = get_item_value(rewards, b, g) if rewards is not None else None
                advantage_value = get_item_value(advantages, b, g) if advantages is not None else None

                # Start response container
                html_output += f'<div class="response-container">'

                # Response header
                html_output += f'<div class="response-header">'
                html_output += f'<div>Response #{g+1}</div>'

                # Add success/fail badge if available
                if is_successful is not None:
                    if is_successful:
                        html_output += f'<div class="success-badge">SUCCESS</div>'
                    else:
                        html_output += f'<div class="failure-badge">FAIL</div>'

                html_output += '</div>'  # End response header

                # Response body with tag highlighting
                html_output += f'<div class="response-body">'

                # Escape HTML but preserve line breaks
                escaped_text = html.escape(response_text).replace('\n', '<br>')

                # Highlight tags
                escaped_text = re.sub(
                    r'&lt;think&gt;(.+?)&lt;/think&gt;',
                    r'<span class="think-tag">&lt;think&gt;</span>\1<span class="think-tag">&lt;/think&gt;</span>',
                    escaped_text,
                    flags=re.DOTALL
                )

                escaped_text = re.sub(
                    r'&lt;answer_era&gt;(.+?)&lt;/answer_era&gt;',
                    r'<span class="answer-era-tag">&lt;answer_era&gt;</span>\1<span class="answer-era-tag">&lt;/answer_era&gt;</span>',
                    escaped_text
                )

                escaped_text = re.sub(
                    r'&lt;answer_date&gt;(.+?)&lt;/answer_date&gt;',
                    r'<span class="answer-date-tag">&lt;answer_date&gt;</span>\1<span class="answer-date-tag">&lt;/answer_date&gt;</span>',
                    escaped_text
                )

                html_output += escaped_text
                html_output += '</div>'  # End response body

                # Metrics container
                if reward_value is not None or advantage_value is not None:
                    html_output += f'<div class="metrics-container">'

                    # Determine score class based on reward value
                    score_class = "score-high" if reward_value and reward_value >= 80 else \
                                "score-medium" if reward_value and reward_value >= 30 else \
                                "score-low"

                    # Display reward
                    if reward_value is not None:
                        html_output += f'<div><strong class="metric-label">Reward:</strong> <span class="metric-score {score_class}">{reward_value:.1f}</span></div>'

                    # Display advantage
                    if advantage_value is not None:
                        adv_class = "score-high" if advantage_value > 0 else "score-low"
                        html_output += f'<div><strong class="metric-label">Advantage:</strong> <span class="metric-score {adv_class}">{advantage_value:.1f}</span></div>'

                    # Add details toggle if available
                    if details is not None and b < len(details) and g < len(details[b]):
                        html_output += f'<a id="toggle-{b}-{g}" class="metrics-toggle" onclick="toggleDetails({b}, {g})">Show Details</a>'
                        html_output += f'<div id="details-{b}-{g}" class="details-container">'

                        # Format reward details
                        detail_data = details[b][g]

                        # Ground truth
                        html_output += f'<div style="margin-bottom: 15px;">'
                        html_output += f'<div><strong>Ground Truth:</strong> Era=\'{detail_data["ground_truth"]["era"]}\', Date=\'{detail_data["ground_truth"]["date"]}\'</div>'
                        html_output += f'</div>'

                        # Extracted tags
                        html_output += f'<div style="margin-bottom: 15px;">'
                        html_output += f'<div><strong>Extracted Tags:</strong></div>'
                        html_output += f'<ul style="margin-top: 5px; padding-left: 20px;">'
                        html_output += f'<li>Think: {len(detail_data["extracted_tags"]["think"])} tag(s)</li>'
                        html_output += f'<li>Era: {len(detail_data["extracted_tags"]["answer_era"])} tag(s)</li>'
                        html_output += f'<li>Date: {len(detail_data["extracted_tags"]["answer_date"])} tag(s)</li>'
                        html_output += f'</ul>'
                        html_output += f'</div>'

                        # Format analysis
                        html_output += f'<div style="margin-bottom: 15px;">'
                        html_output += f'<div><strong>Format Analysis:</strong></div>'

                        if detail_data['format_analysis'].get('has_outside_text', False):
                            outside_text = html.escape(detail_data['format_analysis']['outside_text'])
                            html_output += f'<div class="check-fail">❌ Text outside tags: "{outside_text}"</div>'
                        else:
                            html_output += f'<div class="check-success">✓ No text outside tags</div>'

                        html_output += f'</div>'

                        # Content analysis
                        if 'content_analysis' in detail_data:
                            html_output += f'<div style="margin-bottom: 15px;">'
                            html_output += f'<div><strong>Content Analysis:</strong></div>'

                            # Era analysis
                            if 'era' in detail_data['content_analysis']:
                                provided_eras = ', '.join([f"'{era}'" for era in detail_data['content_analysis']['era']['provided']])
                                html_output += f'<div style="margin-top: 5px;"><strong>Era provided:</strong> {provided_eras}</div>'

                                match_status = ""
                                if detail_data['content_analysis'].get('era_match', {}).get('exact_match', False):
                                    match_status = f'<span class="check-success">✓ Exact match</span>'
                                elif detail_data['content_analysis'].get('era_match', {}).get('partial_match', False):
                                    match_status = f'<span style="color: #BE6A1A; font-weight: bold;">~ Partial match</span>'
                                else:
                                    match_status = f'<span class="check-fail">❌ No match</span>'

                                html_output += f'<div><strong>Era match:</strong> {match_status}</div>'

                            # Date analysis
                            if 'date' in detail_data['content_analysis']:
                                provided_dates = ', '.join([f"'{date}'" for date in detail_data['content_analysis']['date']['provided']])
                                html_output += f'<div style="margin-top: 5px;"><strong>Date provided:</strong> {provided_dates}</div>'

                                if 'best_match' in detail_data['content_analysis']['date']:
                                    best = detail_data['content_analysis']['date']['best_match']
                                    diff_class = "check-success" if best['difference'] <= 20 else \
                                                ("color: #BE6A1A; font-weight: bold;" if best['difference'] <= 50 else "check-fail")

                                    html_output += f'<div><strong>Best date:</strong> {best["value"]} <span style="{diff_class}">(diff: {best["difference"]} years)</span></div>'

                            html_output += f'</div>'

                        # Reward components table
                        html_output += f'<div style="margin-bottom: 15px;">'
                        html_output += f'<div><strong>Reward Components:</strong></div>'
                        html_output += f'<div style="margin-top: 10px;">'

                        for component in detail_data['reward_components']:
                            component_name = component['component']
                            value = component['value']
                            reason = component['reason']

                            # Determine CSS class based on value
                            try:
                                value_float = float(str(value).replace("(overwrites previous)", ""))
                                value_class = "component-value-positive" if value_float > 0 else "component-value-negative"
                            except:
                                value_class = ""

                            html_output += f'<div class="reward-component">'
                            html_output += f'<div><span class="component-name">{component_name}:</span> <span class="component-value {value_class}">{value}</span></div>'
                            html_output += f'<div class="component-reason">{reason}</div>'
                            html_output += f'</div>'

                        html_output += f'</div>'
                        html_output += f'</div>'

                        # Success criteria
                        if 'success_criteria' in detail_data:
                            criteria = detail_data['success_criteria']
                            html_output += f'<div style="margin-bottom: 15px;">'
                            html_output += f'<div><strong>Success Criteria:</strong></div>'
                            html_output += f'<ul style="margin-top: 5px; padding-left: 20px;">'

                            html_output += f'<li><span class="{"check-success" if criteria["perfect_format"] else "check-fail"}">{("✓" if criteria["perfect_format"] else "❌")} Format perfect</span></li>'
                            html_output += f'<li><span class="{"check-success" if criteria["correct_era"] else "check-fail"}">{("✓" if criteria["correct_era"] else "❌")} Era correct</span></li>'
                            html_output += f'<li><span class="{"check-success" if criteria["correct_date"] else "check-fail"}">{("✓" if criteria["correct_date"] else "❌")} Date correct</span></li>'

                            html_output += f'</ul>'
                            html_output += f'</div>'

                        # Total reward summary
                        html_output += f'<div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid #C4C7AC;">'
                        html_output += f'<div><strong>Total Reward:</strong> <span class="{"score-high" if detail_data["total_reward"] > 0 else "score-low"}">{detail_data["total_reward"]}</span></div>'
                        html_output += f'<div><strong>Success:</strong> <span class="{"score-high" if detail_data["success"] > 0 else "score-low"}">{detail_data["success"]}</span></div>'
                        html_output += f'</div>'

                        html_output += f'</div>'  # End details container

                    html_output += f'</div>'  # End metrics container

                html_output += f'</div>'  # End response container

        return html_output


