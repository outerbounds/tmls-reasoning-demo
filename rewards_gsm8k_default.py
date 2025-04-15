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


    def shaped_correctness_reward(self, answer: str, completion: str) -> tuple[float, float]:
        """
        Reward function for verifiable rewards with some mild shaping.

        Args:
            answer (str): ground-truth answer to the current problem
            completion (str): model's completion, starting immediately after "Assistant: <think>"
        Returns:
            reward: (float) a shaped reward indicating the correct answer and the correct format
            success: (float) a binary measure of success (1 if the answer is correct and correctly formatted, 0 otherwise)
        """
        reward = 0.0
        success = 0.0

        try:
            tags = self.extract_tags("<think>" + completion.replace("<<", "").replace(">>", ""))
        except ET.ParseError:
            tags = {"think": [], "answer": []}

        if len(tags["answer"]) == 1:
            reward += 5.0

        if len(tags["think"]) == 1:
            reward += 5.0

        if any(attempt == answer for attempt in tags["answer"]):
            # One of the answer tags has the right answer
            reward += 20.0

        if any((answer in attempt) for attempt in tags["answer"]):
            # One of the answer tags contains the right answer (might be e.g. $20 instead of 20)
            reward += 10.0

        if len(tags["answer"]) > 0 and tags["answer"][-1] == answer:
            reward = 100.0
            success = 1

        return reward, success


    def batch_shaped_correctness_reward(
        self, tokenizer: ModelTokenizer, completions: torch.Tensor, answers: list[str]
    ) -> Tuple[torch.Tensor]:
        """Utility function to apply the shaped reward function to a GRPO-style batch of completions."""

        batch_size, grpo_size, *_ = completions.shape
        rewards = torch.zeros(batch_size, grpo_size, dtype=torch.float32)
        successes = torch.zeros(batch_size, grpo_size, dtype=torch.float32)
        # completions :: [B, G, L]
        for b in range(batch_size):
            for g in range(grpo_size):
                text_completion = tokenizer.decode(
                    completions[b, g].tolist()
                )  # skips special tokens, stops at eos
                reward, success = self.shaped_correctness_reward(
                    answer=answers[b], completion=text_completion
                )
                rewards[b, g] = reward
                successes[b, g] = success

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

                        if 'logic_analysis' in detail_data:
                            # model, premise, hypothesis, inference
                            html_output += f'<div style="margin-bottom: 15px;">'
                            html_output += f'<div><strong>Logic Analysis:</strong></div>'
                            
                            model = detail_data['logic_analysis'].get('model', None)
                            premise = detail_data['logic_analysis'].get('premise', None)
                            hypothesis = detail_data['logic_analysis'].get('hypothesis', None)
                            inference = detail_data['logic_analysis'].get('inference', None)

                            if model:
                                html_output += f'<div style="margin-top: 10px;"><strong>Model:</strong> {model}</div>'

                            if premise:
                                html_output += f'<div style="margin-top: 10px;">'
                                html_output += f'<div><strong>Premise:</strong> <span style="font-style: italic; color: #555; font-size: 0.9em;">(content from &lt;think&gt; tag)</span></div>'
                                html_output += f'<div style="background-color: #f8f9fa; padding: 8px; border-radius: 4px; margin-top: 5px; font-size: 0.9em; border-left: 3px solid #BE6A1A;">'
                                html_output += html.escape(premise).replace('\n', '<br>')
                                html_output += f'</div>'
                                html_output += f'</div>'

                            if hypothesis:
                                html_output += f'<div style="margin-top: 10px;">'
                                html_output += f'<div><strong>Hypothesis:</strong> <span style="font-style: italic; color: #555; font-size: 0.9em;">(derived from answers)</span></div>'
                                html_output += f'<div style="background-color: #f8f9fa; padding: 8px; border-radius: 4px; margin-top: 5px; font-size: 0.9em; border-left: 3px solid #2C6846;">'
                                html_output += html.escape(hypothesis)
                                html_output += f'</div>'
                                html_output += f'</div>'

                            if inference:
                                html_output += f'<div style="margin-top: 10px;"><strong>Judgement:</strong> '
                                
                                # Apply appropriate styling based on the inference result
                                if inference == "Entailment":
                                    html_output += f'<span class="check-success">✓ {inference}</span>'
                                    html_output += f' <span style="font-size: 0.85em;">(The reasoning supports the answers)</span>'
                                elif inference == "Neutral":
                                    html_output += f'<span style="color: #BE6A1A; font-weight: bold;">⚠ {inference}</span>'
                                    html_output += f' <span style="font-size: 0.85em;">(The reasoning neither supports nor contradicts the answers)</span>'
                                elif inference == "Contradiction":
                                    html_output += f'<span class="check-fail">❌ {inference}</span>'
                                    html_output += f' <span style="font-size: 0.85em;">(The reasoning contradicts the answers)</span>'
                                else:
                                    html_output += f'<span style="color: #BE6A1A; font-weight: bold;">? {inference}</span>'
                                
                                html_output += f'</div>'

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
                        
                        html_output += f'</div>'  
                    ## END DETAILS CONTAINER ##
                    
                    html_output += f'</div>'  
                ## END FOLDABLE METRICS CONTAINER ##
                
                html_output += f'</div>'  
            ## END RESPONSE CONTAINER ##
        
        return html_output