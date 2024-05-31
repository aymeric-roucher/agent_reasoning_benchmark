from manim import *

class TextExample(Scene):
    def construct(self):
        text = Text("Here is a text", font="Consolas", font_size=90)
        self.play(Write(text))
        self.wait(3)

SCALE = 0.4

rescaling_factor = 2


class Agent(Scene):
    def animate_pulsing_effect(self, arrow):
        self.play(arrow.animate.scale(rescaling_factor), run_time=0.3)
        self.play(arrow.animate.scale(1 / rescaling_factor), run_time=0.3)
        self.wait(0.5)

    def construct(self):
        # Title
        # title = Text("Agent - Multi step/ ReAct", font_size=32, color=RED).to_edge(UP)
        # self.play(Write(title))
        Text.set_default(font="Consolas", font_size=30)
        

        # Step 1: System prompt template
        system_prompt_template_box = Rectangle(width=20, height=2.5, color=WHITE).scale(SCALE)
        system_prompt_template_text = Text(
            'System prompt:\n"Solve this task in an iterative way with a Thought/Action/Observation loop.\nYou can use these tools: ["calculator", "web_search"]\nTask: how much is 2 + 2?"',
        ).scale(SCALE).move_to(system_prompt_template_box.get_center())
        system_prompt_template_group = VGroup(system_prompt_template_box, system_prompt_template_text).to_edge(UP)
        self.play(FadeIn(system_prompt_template_group))
        self.wait(0.3)
        
        # Step 2: Initialize memory and prompt
        memory_box = Rectangle(width=11, height=3.5, color=WHITE).scale(SCALE)
        memory_text = Text("Memory: []").scale(SCALE).move_to(memory_box.get_center())
        memory_group = VGroup(memory_box, memory_text).next_to(system_prompt_template_group, DOWN, buff=0)
        
        prompt_box = Rectangle(width=11, height=1, color=WHITE).scale(SCALE)
        prompt_text = Text("Prompt = System prompt + Memory").scale(SCALE).move_to(prompt_box.get_center())
        prompt_group = VGroup(prompt_box, prompt_text).next_to(memory_group, DOWN, buff=0)
        
        self.play(FadeIn(memory_box), Write(memory_text), FadeIn(prompt_box), Write(prompt_text))
        
        # Step 3: Call LLM
        call_llm_box = Rectangle(width=3, height=1, color=ORANGE).scale(SCALE)
        call_llm_text = Text("Run LLM").scale(SCALE).move_to(call_llm_box.get_center())
        call_llm_group = VGroup(call_llm_box, call_llm_text).next_to(prompt_group, DOWN*SCALE, buff=1)
        
        
        arrow1 = Arrow(start=prompt_group.get_bottom(), end=call_llm_group.get_top(), buff=0.1)
        self.play(FadeIn(arrow1), FadeIn(call_llm_box), Write(call_llm_text))

        # Step 3.5: LLM Output
        llm_output_box = Rectangle(width=10, height=2, color=WHITE).scale(SCALE)
        llm_output_text = Text("LLM output:\nThought: I should use the calculator.\nAction: calculator(2+2)").scale(SCALE).move_to(llm_output_box.get_center())
        llm_output_group = VGroup(llm_output_box, llm_output_text).next_to(call_llm_group, DOWN*SCALE, buff=1)
        
        arrow1_5 = Arrow(start=call_llm_group.get_bottom(), end=llm_output_group.get_top(), buff=0.1)
        self.play(FadeIn(arrow1_5), FadeIn(llm_output_box), Write(llm_output_text))
    
        # Step 4: Parse tool call(s) from output
        parse_box = Rectangle(width=9, height=1, color=BLUE).scale(SCALE)
        parse_text = Text("Parse tool call(s) from output").scale(SCALE).move_to(parse_box.get_center())
        parse_group = VGroup(parse_box, parse_text).next_to(llm_output_group, DOWN*SCALE, buff=1)
        
        arrow2 = Arrow(start=llm_output_group.get_bottom(), end=parse_group.get_top(), buff=0.1)
        self.play(FadeIn(arrow2), FadeIn(parse_box), Write(parse_text))
        
        # Step 5: Resulting tool call(s)
        parsed_tool_call_box = Rectangle(width=6, height=1.5, color=WHITE).scale(SCALE)
        parsed_tool_call_text = Text("Tool calls:\ncalculator(2 + 2)").scale(SCALE).move_to(parsed_tool_call_box.get_center())
        parsed_tool_call_group = VGroup(parsed_tool_call_box, parsed_tool_call_text).next_to(parse_group, DOWN*SCALE, buff=1)

        arrow3 = Arrow(start=parse_group.get_bottom(), end=parsed_tool_call_group.get_top(), buff=0.1)
        self.play(FadeIn(arrow3), FadeIn(parsed_tool_call_box), Write(parsed_tool_call_text))


        # Step 6: Decision
        arrow_decision_no = Arrow(start=parsed_tool_call_group.get_right(), end=parsed_tool_call_group.get_right() + RIGHT*2, buff=0.1, tip_length=0.15)
        no_text = Text("Normal tool call", color=BLUE).scale(SCALE).next_to(arrow_decision_no, UP*SCALE*0.5)
        self.play(FadeIn(arrow_decision_no), Write(no_text))
        
        # Execute call
        tool_call_box = Rectangle(width=5, height=1, color=BLUE).scale(SCALE)
        tool_call_text = Text("Execute call").scale(SCALE).move_to(tool_call_box.get_center())
        tool_call_group = VGroup(tool_call_box, tool_call_text).next_to(arrow_decision_no.get_end(), RIGHT)

        self.play(FadeIn(tool_call_box), Write(tool_call_text))


        # Adding "Observation" text
        observation_text = Text("Observation: 4", color=YELLOW).scale(SCALE)
        observation_text.move_to(tool_call_group.get_top() + UP * 0.5)

        # Moving "Observation" text along the CurvedArrow
        loop_arrow = CurvedArrow(start_point=tool_call_group.get_top(), end_point=memory_group.get_right() + RIGHT*SCALE, angle=TAU/4, tip_length=0.15)
        self.play(FadeIn(observation_text))
        self.play(FadeIn(loop_arrow), MoveAlongPath(observation_text, loop_arrow), run_time=2)

        # Adding the "Observation" text to the memory
        updated_memory_text = Text('memory = [\n(Last LLM output + Observation)\n]').scale(SCALE).move_to(memory_box.get_center())
        self.play(Transform(memory_text, updated_memory_text), FadeOut(observation_text))
        self.wait(0.5)

        updated_memory_text = Text('memory = [\n"LLM output:\nThought: I should use the calculator.\nAction: calculator(2+2)\nObservation: 4"\n]').scale(SCALE).move_to(memory_box.get_center())
        self.play(Transform(memory_text, updated_memory_text))
        self.wait(0.5)

        # Going again through the loop
        # Create a pulsing effect
        self.animate_pulsing_effect(arrow1)
        self.animate_pulsing_effect(arrow1_5)

        # Update LLM output:
        updated_llm_output_text = Text("LLM output:\nThought: I should return the result.\nAction: final_answer(4)").scale(SCALE).move_to(llm_output_box.get_center())
        self.play(Transform(llm_output_text, updated_llm_output_text))

        self.animate_pulsing_effect(arrow2)

        updated_parsed_tool_call_text = Text("Tool calls:\nfinal_answer(4)").scale(SCALE).move_to(parsed_tool_call_box.get_center())
        self.play(Transform(parsed_tool_call_text, updated_parsed_tool_call_text))
                  
        # Return result
        arrow_decision_yes = Arrow(start=parsed_tool_call_group.get_bottom(), end=parsed_tool_call_group.get_bottom() + DOWN*SCALE, buff=0.1)

        final_text = Text("Final answer", color=GREEN).scale(SCALE).next_to(arrow_decision_yes, RIGHT)
        self.play(FadeIn(arrow_decision_yes), Write(final_text))

        # Return result
        result_text = Text("Return result: 4").scale(SCALE).next_to(arrow_decision_yes.get_end(), DOWN*SCALE)
        self.play(Write(result_text))
        self.wait(2)
