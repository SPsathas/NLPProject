import json
import openai

class DevDataset:
    """
    a class for handling the dev.json dataset.

    Parameters:
    path (str): the path to the dev.json file.

    Attributes:
    questions (list): a list of Question objects, each representing one question of dev.json.
    """

    def __init__(self, path: str):
        """
        initializes a DevDataset object.

        Args:
        path (str): path to the dataset file.
        """
        with open(path, 'r', encoding='cp850') as file:
            data = json.load(file)


        self.questions = []
        self.lut = {}

        for index, q in enumerate(data):
            question = Question(q)
            self.questions.append(question)
            self.lut[question.id] = index

    def __len__(self):
        """
        returns the length of the dataset.

        Returns:
        int: number of questions in the dataset.
        """
        return len(self.questions)
    
    def __getitem__(self, key: int | str):
        """
        returns the Question object located at the specified index or by id.

        Args:
        key (int | str): the index or id of the question to retrieve.

        Returns:
        Question: the question at the specified index or id.
        """
        if isinstance(key, int):
            return self.questions[key]
        elif isinstance(key, str):
            index = self.lut.get(key)
            if index is not None:
                return self.questions[index]
            else:
                raise KeyError(f"No question found with ID: {key}")
    
    def __iter__(self):
        """
        returns an iterator over the questions in the dataset.

        Returns:
        iter: an iterator over the questions in the dataset.
        """
        return iter(self.questions)
        

class Question:
    """
    represents one question with its associated contexts, supporting facts, evidences, and answer...

    Attributes:
        id (str): the id of the question.
        type (str): the type of the question.
        question (str): the text of the question.
        contexts (list): a list of contexts associated with the question.
        gold_contexts (list): a list of gold contexts associated with the question.
        other_contexts (list): a list of other contexts associated with the question.
        supporting_facts (list): a list of supporting facts associated with the question.
        evidences (list): a list of evidences associated with the question.
        answer (str): the answer to the question.
    """

    def __init__(self, data: dict):
        """
        constructs a Question object.

        Args:
            data (dict): a dictionary containing the data of the question from the .json file.
        """

        if not isinstance(data, dict):
            raise TypeError('data must be a dictionary.')

        self.id = data['_id']
        self.type = data['type']
        self.question = data['question']
        self.contexts = data['context']
        self.gold_contexts = []
        self.other_contexts = []
        self.supporting_facts = data['supporting_facts']
        self.evidences = data['evidences']
        self.answer = data['answer']

        for context in self.contexts:
            if context[0] in [supporting_fact[0] for supporting_fact in self.supporting_facts]:
                self.gold_contexts.append(context)
            else:
                self.other_contexts.append(context)
    
    def get_contexts(self, context_type='gold', include_titles=True):
        """
        returns a list of contexts associated with the question, read from the .json file.

        Args:
            context_type (str, optional): the type of contexts to return. Can be 'gold', 'other', or 'all'. Defaults to 'gold'.
            include_titles (bool, optional): whether to include the titles of the contexts. Defaults to True.

        Returns:
            str: a list of context strings. has the structure:
                    [
                        '<titleA>:\n<paragraphA1>\n<paragraphA2>\n...',
                        '<titleB>:\n<paragraphB1>\n<paragraphB2>\n...',
                        ...
                    ]
                or:
                    [
                        '<paragraphA1>\n<paragraphA2>\n...',
                        '<paragraphB1>\n<paragraphB2>\n...',
                        ...
                    ]
        """

        if not isinstance(context_type, str):
            raise TypeError('context_type must be a string.')
        if context_type not in ['gold', 'other', 'all']:
            raise ValueError('context_type must be either "gold", "other", or "all".')
        if not isinstance(include_titles, bool):
            raise TypeError('include_titles must be a boolean.')

        def build_context_list(context_list, include_titles):
            contexts = []
            for context in context_list:
                string = ''
                if include_titles:
                    string += context[0] + ':\n'
                for i_par, par in enumerate(context[1]):
                    string += par
                    string += '\n' if i_par < len(context[1]) - 1 else ''
                contexts.append(string)
            return contexts
        
        match context_type:
            case 'gold':
                return build_context_list(self.gold_contexts, include_titles)
            case 'other':
                return build_context_list(self.other_contexts, include_titles)
            case _:
                return build_context_list(self.contexts, include_titles)
            
class GPTQA:
    def __init__(self):
        """
        initialize a GPTQA object.

        Args:
            api_key (str): The OpenAI API key.
        """
        self.client = openai.Client(api_key="sk-proj-OD9UpwZjabyMJ0bPtPnMT3BlbkFJKDkmftX3AjUy6zp6di0P")
        self.system_prompt = 'Respond in the shortest way possible. No full sentence. Just a factoid answer.'
        self.actually_prompt = True

    def prompt_gpt(self, prompt: str) -> str:
        """
        prompt gpt-3.5 with a question and return the generated answer.

        Args:
            prompt (str): the string to prompt gpt-3.5 with.

        Returns:
            str: a string containing the generated answer.
        """

        if not isinstance(prompt, str):
            raise TypeError('prompt must be a string.')
        if not isinstance(self.system_prompt, str):
            raise TypeError('system_prompt must be a string.')

        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content

    def ask_question(self, question: str|Question, context: str|list[str] = None) -> tuple[str, str] | tuple[str, str, str]:
        """
        ask a question to the GPT-3.5 model and return the response.

        Args:
            question (str|Question): the question to ask.
            context (str|list[str], optional): the context to provide with the question. Default None.

        Returns:
            str: the response to the question.
        """

        if isinstance(question, Question):
            q = question.question
        elif isinstance(question, str):
            q = question
        else:
            raise TypeError('question must be a string or a Question object.')
        if not isinstance(context, (str, list, type(None))):
            raise TypeError('context must be a string, a list of strings or None.')
        
        context_str = ''

        if context is not None:
            if isinstance(context, list):
                for c in context:
                    context_str += c + '\n\n'
            elif isinstance(context, str):
                context_str = context + '\n\n'
            else:
                raise TypeError('context must be a string or a list of strings.')
        
        prompt = context_str + q
        
        if self.actually_prompt:
            response = self.prompt_gpt(prompt)
            if isinstance(question, Question):
                return (prompt, response, question.answer)
            else:
                return (prompt, response)
        else:
            return (prompt, 'dummy response', question.answer)
    
    def set_system_prompt(self, prompt: str):
        """
        set the system prompt for the GPT-3.5 chat completion.

        Args:
            prompt (str): the new system prompt.
        """

        if not isinstance(prompt, str):
            raise TypeError('prompt must be a string.')
        self.system_prompt = prompt

    def print_response(self, response: tuple[str, str]|tuple[str, str, str]):
        """
        print the question, response and ground truth to the console.

        Args:
            response (tuple[str, str]|tuple[str, str, str]): the response to print.
        """

        if isinstance(response, str):
            print(response)
        elif isinstance(response, tuple):
            print('_____Prompt_____')
            print(response[0] + '\n')
            print('_____Response____')
            print(response[1])
            if len(response) > 2:
                print('\n_____Ground Truth_____')
                print(response[2])
        else:
            raise TypeError('response must be a string or a tuple of strings.')
        