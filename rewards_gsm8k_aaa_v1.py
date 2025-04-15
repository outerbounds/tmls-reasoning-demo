# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from xml.etree import ElementTree as ET

import torch
from typing import Optional, List, Dict, Any, Tuple
import re 
import html

from torchtune.modules.transforms.tokenizers import ModelTokenizer

class RewardServer(object):

    def extract_tags(self, text: str) -> dict[str, list[str]]:
        """
        Parse XML-like tags from text. Returns a dictionary with keys 'think' and 'answer'.
        The values are lists of strings, with each string being the content of a tag.
        """
        xml_string = f"<root>{text}</root>"
        root = ET.fromstring(xml_string)

        return {
            "think": [
                elem.text if elem.text is not None else "" for elem in root.findall("think")
            ],
            "answer": [
                elem.text if elem.text is not None else ""
                for elem in root.findall("answer")
            ],
        }


    def shaped_correctness_reward(self, answer: str, completion: str) -> tuple[float, float, list]:
        """
        A more complex reward function that encourages discovery of interesting patterns.
        
        Args:
            answer (str): ground-truth answer to the current problem
            completion (str): model's completion
        Returns:
            reward: (float) a shaped reward 
            success: (float) a binary measure of success (1 if fully successful, 0 otherwise)
            details: (list) explanation of rewards given

        https://claude.ai/chat/be3991b9-1127-4e13-92c4-40821c0335f1
        """
        reward = 0.0
        success = 0.0
        details = {
            "completion": completion,
            "extracted_tags": {},
            "format_analysis": {},
            "content_analysis": {},
            "reward_components": [],
            "total_reward": 0.0,
            "success": 0.0
        }
        
        try:
            tags = self.extract_tags("<think>" + completion.replace("<<", "").replace(">>", ""))
        except ET.ParseError:
            tags = {"think": [], "answer": []}
        details["extracted_tags"] = {
            "think": tags["think"],
            "answer": tags["answer"]
        }

        ### FORMATTING PENALTY ###
        
        ## <think> FORMATTING ## 
        # Max reward: 5.
        # Min reward: 0.
        if len(tags["think"]) == 1:
            reward += 5.0
            details["reward_components"].append({
                "component": "think_tag_format",
                "value": 5.0,
                "reason": "Correct: Exactly one <think> tag"
            })
        
        ## <answer> FORMATTING ## 
        # Max reward: 5.
        # Min reward: 0.
        if len(tags["answer"]) == 1:
            reward += 5.0
            details["reward_components"].append({
                "component": "answer_tag_format",
                "value": 5.0,
                "reason": "Correct: Exactly one <answer> tag"
            })
        
        # If no valid answer tag, no further rewards
        if len(tags["answer"]) == 0:
            return reward, success, details
        
        answer_text = tags["answer"][0].lower()
        think_text = tags["think"][0].lower() if len(tags["think"]) > 0 else ""
        
        # <answer> pattern discovery rewards
        # Max reward: 35.
        # Min reward: 
        # Reward 1: Symmetry - reward answers that read the same forward and backward
        if answer_text == answer_text[::-1] and len(answer_text) >= 3:
            palindrome_reward = 30.0
            reward += palindrome_reward
            details["reward_components"].append({
                "component": "palindrome_pattern",
                "value": 30.,
                "reason": "The answer is a palindrome-like pattern."
            })
            
            # Extra reward for longer palindromes, up to a limit
            if len(answer_text) > 3:
                length_bonus = min(len(answer_text) - 3, 7) * 5.0  # Cap at +35 points
                reward += length_bonus
                details["reward_components"].append({
                    "component": "palindrome_length_bonus",
                    "value": length_bonus,
                    "reason": f"The palindrome-like pattern is {len(answer_text)}."
                })
        
        # Reward 2: Pattern discovery - reward repeated patterns
        for pattern_length in range(1, 4):  # Look for patterns of length 1-3
            if len(answer_text) >= pattern_length * 2:
                pattern = answer_text[:pattern_length]
                repetitions = 1
                
                # Count how many times the pattern repeats from the beginning
                for i in range(pattern_length, len(answer_text), pattern_length):
                    if i + pattern_length <= len(answer_text) and answer_text[i:i+pattern_length] == pattern:
                        repetitions += 1
                    else:
                        break
                
                if repetitions >= 2:
                    pattern_reward = 15.0 * repetitions
                    reward += pattern_reward
                    details["reward_components"].append({
                        "component": "repeated_patterns",
                        "value": pattern_reward,
                        "reason": f"The text contains repeated patterns."
                    })
            
        # Reward 3: Reasoning alignment - reward if the answer logically follows from thinking
        if len(think_text) > 0 and "aaa" in think_text and "aaa" in answer_text:
            reward += 25.0
            # details.append('reasoning:aligned+25')
            details["reward_components"].append({
                "component": "reasoning_alignment_aaa",
                "value": 25.,
                "reason": f"The text contains 'aaa' in <think> and <answer>."
            })
        
        # Success criteria: Perfect answers
        # Define success for exact "aaa" or perfect palindromes of length >= 5
        if answer_text == "aaa":
            reward = 100.0
            success = 1.0
            # details = ['success:aaa']
            details["reward_components"].append({
                "component": "perfect_answer",
                "value": "100.0 (overwrites previous)",
                "reason": "Perfect answer."
            })
        elif answer_text == answer_text[::-1] and len(answer_text) >= 5:
            reward = 100.0
            success = 1.0
            # details = [f'success:palindrome_{answer_text}']
            details["reward_components"].append({
                "component": "perfect_palindrome",
                "value": "100.0 (overwrites previous)",
                "reason": "Perfect palindrome-ish response."
            })
        
        # Length normalization - penalize very long answers
        if len(answer_text) > 10:
            penalty = min(0.5, (len(answer_text) - 10) * 0.05)  # Up to 50% penalty
            reward *= (1 - penalty)
            # details.append(f'penalty:length-{penalty*100}%')
            details["reward_components"].append({
                "component": "length_penalty",
                "value":  - penalty * reward,
                "reason": "Penalty for long answers."
            })
        
        return reward, success, details


    def batch_shaped_correctness_reward(
        self, 
        tokenizer: ModelTokenizer, 
        completions: torch.Tensor, 
        answers: list[str],
        details_report: bool = False
    ) -> [torch.Tensor, torch.Tensor]:
        """Utility function to apply the shaped reward function to a GRPO-style batch of completions."""

        batch_size, grpo_size, *_ = completions.shape
        rewards = torch.zeros(batch_size, grpo_size, dtype=torch.float32)
        successes = torch.zeros(batch_size, grpo_size, dtype=torch.float32)

        # Create container for details if requested
        details_list = [] if details_report else None

        # completions :: [B, G, L]
        for b in range(batch_size):
            batch_details = [] if details_report else None

            for g in range(grpo_size):
                text_completion = tokenizer.decode(
                    completions[b, g].tolist()
                )  # skips special tokens, stops at eos
                reward, success, details = self.shaped_correctness_reward(
                    answer=answers[b], completion=text_completion
                )
                rewards[b, g] = reward
                successes[b, g] = success

                if details_report:
                    # Add batch and group indices
                    details["batch_idx"] = b
                    details["group_idx"] = g
                    batch_details.append(details)
            
            if details_report:
                details_list.append(batch_details)

        if details_report:
            return rewards, successes, details_list
        else:
            return rewards, successes

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
    ):
        """
        Display responses with rewards, advantages, and detailed diagnostics in a visually appealing format.
        Specifically adapted for the shaped_correctness_reward function.
        
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
            .pattern-highlight {
                background-color: #f8f9fa;
                padding: 8px;
                border-radius: 4px;
                margin-top: 5px;
                font-size: 0.9em;
                border-left: 3px solid #2C6846;
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
                response_text = tokenizer.decode(responses[b, g].tolist())
                success_value = get_item_value(successes, b, g) if successes is not None else None
                is_successful = success_value == 1.0 if success_value is not None else None
                reward_value = get_item_value(rewards, b, g) if rewards is not None else None
                advantage_value = get_item_value(advantages, b, g) if advantages is not None else None
                
                # Start response container
                html_output += f'<div class="response-container">'
                
                ## START RESPONSE HEADER ##
                html_output += f'<div class="response-header">'
                html_output += f'<div>Response #{g+1}</div>'
                
                # Add success/fail badge if available
                if is_successful is not None:
                    if is_successful:
                        html_output += f'<div class="success-badge">SUCCESS</div>'
                    else:
                        html_output += f'<div class="failure-badge">FAIL</div>'
                
                html_output += '</div>'  
                ## END RESPONSE HEADER ##
                
                ## START RESPONSE BODY ##
                html_output += f'<div class="response-body">'
                
                # Escape HTML but preserve line breaks
                escaped_text = html.escape(response_text).replace('\n', '<br>')

                # Highlight <think>, <answer> tags
                escaped_text = re.sub(
                    r'&lt;think&gt;(.+?)&lt;/think&gt;',
                    r'<span class="think-tag">&lt;think&gt;</span>\1<span class="think-tag">&lt;/think&gt;</span>',
                    escaped_text,
                    flags=re.DOTALL
                )
                
                escaped_text = re.sub(
                    r'&lt;answer&gt;(.+?)&lt;/answer&gt;',
                    r'<span class="answer-tag">&lt;answer&gt;</span>\1<span class="answer-tag">&lt;/answer&gt;</span>',
                    escaped_text,
                    flags=re.DOTALL
                )
                
                html_output += escaped_text
                html_output += '</div>'  
                ## END RESPONSE BODY ##
                
                ## START FOLDABLE METRICS CONTAINER ##
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
                    
                    ## START DETAILS CONTAINER ##
                    if details is not None and b < len(details) and g < len(details[b]):
                        html_output += f'<a id="toggle-{b}-{g}" class="metrics-toggle" onclick="toggleDetails({b}, {g})">Show Details</a>'
                        html_output += f'<div id="details-{b}-{g}" class="details-container">'
                        
                        # Format reward details
                        detail_data = details[b][g]
                        
                        # Extracted tags
                        html_output += f'<div style="margin-bottom: 15px;">'
                        html_output += f'<div><strong>Extracted Tags:</strong></div>'
                        html_output += f'<ul style="margin-top: 5px; padding-left: 20px;">'
                        html_output += f'<li>Think: {len(detail_data["extracted_tags"]["think"])} tag(s)</li>'
                        html_output += f'<li>Answer: {len(detail_data["extracted_tags"]["answer"])} tag(s)</li>'
                        html_output += f'</ul>'
                        html_output += f'</div>'
                        
                        # Display answer content analysis for palindromes and patterns
                        if "extracted_tags" in detail_data and "answer" in detail_data["extracted_tags"] and len(detail_data["extracted_tags"]["answer"]) > 0:
                            answer_text = detail_data["extracted_tags"]["answer"][0].lower()
                            
                            html_output += f'<div style="margin-bottom: 15px;">'
                            html_output += f'<div><strong>Answer Analysis:</strong></div>'
                            
                            # Is it a palindrome?
                            is_palindrome = answer_text == answer_text[::-1] and len(answer_text) >= 3
                            html_output += f'<div style="margin-top: 10px;">'
                            if is_palindrome:
                                html_output += f'<div class="check-success">✓ Palindrome detected</div>'
                                html_output += f'<div style="font-size: 0.9em; margin-top: 5px;">Length: {len(answer_text)} characters</div>'
                            else:
                                html_output += f'<div class="check-fail">❌ Not a palindrome</div>'
                            html_output += f'</div>'
                            
                            # Pattern analysis
                            html_output += f'<div style="margin-top: 10px;">'
                            html_output += f'<div><strong>Pattern Analysis:</strong></div>'
                            
                            pattern_found = False
                            for pattern_length in range(1, 4):
                                if len(answer_text) >= pattern_length * 2:
                                    pattern = answer_text[:pattern_length]
                                    repetitions = 1
                                    
                                    # Count how many times the pattern repeats from the beginning
                                    for i in range(pattern_length, len(answer_text), pattern_length):
                                        if i + pattern_length <= len(answer_text) and answer_text[i:i+pattern_length] == pattern:
                                            repetitions += 1
                                        else:
                                            break
                                    
                                    if repetitions >= 2:
                                        pattern_found = True
                                        html_output += f'<div class="check-success">✓ Repeated pattern detected</div>'
                                        html_output += f'<div style="font-size: 0.9em; margin-top: 5px;">Pattern: "{pattern}" repeats {repetitions} times</div>'
                            
                            if not pattern_found:
                                html_output += f'<div class="check-fail">❌ No repeated patterns found</div>'
                            
                            html_output += f'</div>'
                            
                            # Check for "aaa" in both think and answer
                            if "think" in detail_data["extracted_tags"] and len(detail_data["extracted_tags"]["think"]) > 0:
                                think_text = detail_data["extracted_tags"]["think"][0].lower()
                                
                                html_output += f'<div style="margin-top: 10px;">'
                                html_output += f'<div><strong>Reasoning Alignment:</strong></div>'
                                
                                if "aaa" in think_text and "aaa" in answer_text:
                                    html_output += f'<div class="check-success">✓ "aaa" appears in both think and answer tags</div>'
                                else:
                                    html_output += f'<div class="check-fail">❌ "aaa" is not aligned between think and answer tags</div>'
                                
                                html_output += f'</div>'
                            
                            # Perfect answer check
                            html_output += f'<div style="margin-top: 10px;">'
                            html_output += f'<div><strong>Perfect Answer Check:</strong></div>'
                            
                            if answer_text == "aaa":
                                html_output += f'<div class="check-success">✓ Perfect answer: "aaa"</div>'
                            elif answer_text == answer_text[::-1] and len(answer_text) >= 5:
                                html_output += f'<div class="check-success">✓ Perfect palindrome of length {len(answer_text)}</div>'
                            else:
                                html_output += f'<div class="check-fail">❌ Not a perfect answer</div>'
                            
                            html_output += f'</div>'
                            
                            # Length check
                            html_output += f'<div style="margin-top: 10px;">'
                            html_output += f'<div><strong>Length Check:</strong></div>'
                            
                            if len(answer_text) > 10:
                                penalty = min(0.5, (len(answer_text) - 10) * 0.05)
                                html_output += f'<div class="check-fail">❌ Answer is too long ({len(answer_text)} characters)</div>'
                                html_output += f'<div style="font-size: 0.9em; margin-top: 5px;">Penalty applied: {penalty*100:.1f}%</div>'
                            else:
                                html_output += f'<div class="check-success">✓ Answer length is good ({len(answer_text)} characters)</div>'
                            
                            html_output += f'</div>'
                            
                            html_output += f'</div>'
                        
                        # Reward components table
                        if 'reward_components' in detail_data:
                            html_output += f'<div style="margin-bottom: 15px;">'
                            html_output += f'<div><strong>Reward Components:</strong></div>'
                            html_output += f'<div style="margin-top: 10px;">'
                            
                            for component in detail_data['reward_components']:
                                component_name = component['component']
                                value = component['value']
                                reason = component['reason']
                                
                                # Determine CSS class based on value
                                try:
                                    # Handle strings like "100.0 (overwrites previous)"
                                    value_str = str(value)
                                    if "overwrites" in value_str:
                                        value_float = float(value_str.split(" ")[0])
                                    else:
                                        value_float = float(value_str)
                                    
                                    value_class = "component-value-positive" if value_float > 0 else "component-value-negative"
                                except:
                                    value_class = ""
                                
                                html_output += f'<div class="reward-component">'
                                html_output += f'<div><span class="component-name">{component_name}:</span> <span class="component-value {value_class}">{value}</span></div>'
                                html_output += f'<div class="component-reason">{reason}</div>'
                                html_output += f'</div>'
                            
                            html_output += f'</div>'
                            html_output += f'</div>'
                        
                        # Total reward summary
                        html_output += f'<div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid #C4C7AC;">'
                        html_output += f'<div><strong>Total Reward:</strong> <span class="metric-score {score_class}">{reward_value:.1f}</span></div>'
                        html_output += f'<div><strong>Success:</strong> <span class="metric-score {"score-high" if is_successful else "score-low"}">{success_value:.1f}</span></div>'
                        html_output += f'</div>'
                        
                        html_output += f'</div>'  
                    ## END DETAILS CONTAINER ##
                    
                    html_output += f'</div>'  
                ## END FOLDABLE METRICS CONTAINER ##
                
                html_output += f'</div>'  
            ## END RESPONSE CONTAINER ##
        
        return html_output