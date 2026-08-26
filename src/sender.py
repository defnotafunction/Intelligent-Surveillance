import yagmail
from dotenv import load_dotenv
import os
from socket import gaierror
import json

load_dotenv()

SRC_DIR = os.path.dirname(os.path.abspath(__file__)) 
BASE_DIR = os.path.dirname(SRC_DIR)

class GmailSender:
    def __init__(self) -> None:
        self._create_missing_folders_and_files()
        self._app_password = os.getenv('GMAIL_APP_PASSWORD')
        self._gmail_account = os.getenv('GMAIL_ACCOUNT')

        self._yag = yagmail.SMTP(self._gmail_account, self._app_password)

        # For collecting emails that weren't able to be sent
        self.emails_to_send_path = os.path.join(BASE_DIR, 'data', 'emaildata', 'emails_to_send.json')
        with open(self.emails_to_send_path, 'r', encoding="utf-8") as r:
            self._emails_to_send = json.load(r)  

    def _create_missing_folders_and_files(self) -> None:
        """Creates folders and files that the GmailSender class needs to use."""

        required_folders = [
            os.path.join(BASE_DIR, 'data', 'emaildata'),
        ]

        required_files = [
            os.path.join(BASE_DIR, 'data', 'emaildata', 'emails_to_send.json')
        ]

        for folder in required_folders:
            os.makedirs(folder, exist_ok=True)

        for file in required_files:
            if not os.path.exists(file):
                with open(file, "w", encoding="utf-8") as w_file:
                    if file[-4:] == 'json':
                        json.dump({}, w_file, indent=4)

    def _save_emails_to_send(self) -> None:
        """Saves the emails_to_send attribute."""
        with open(self.emails_to_send_path, "w", encoding="utf-8") as w_file:
            json.dump(self._emails_to_send, w_file, indent=4)

    def send(self, email_subject: str, message: str, file_paths: list) -> None | dict:
        """
        Uses yagmail to send an email to and from the account specified in an environment file as 'GMAIL_ACCOUNT'.

        Args:
            email_subject: A string including text that'll be a part of the email as the title line.
            message: A string including text that'll be a part of the email.
            file_paths: A list that contains file paths that'll be a part of the email.
        
        Returns:
            Only returns a dictionary containing the content of the email if yagmail fails to send the email.
        """
        try:
            self._yag.send(
                to=self._gmail_account,
                subject=f'Intelligent Surveillance: {email_subject}',
                contents=[message],
                attachments=file_paths

            )
        except gaierror:  # Exception that is raised when there is no internet connection
            base_key_name = 'emailpending'

            if len(self._emails_to_send) > 0:
                sorted_keys = sorted(list(self._emails_to_send.keys()))
                latest_key_int = int(sorted_keys[-1][-1])
                new_key_name = f'{base_key_name}{latest_key_int + 1}'
            else:
                new_key_name = f'{base_key_name}1'

            self._emails_to_send[new_key_name] = {
                'subject': email_subject,
                'contents': message,
                'attachments': file_paths,
            }

            with open(self.emails_to_send_path, 'w') as w_file:
                json.dump(self._emails_to_send, w_file, indent=4)

    def send_pending_emails(self) -> None:
        """Uses yagmail to attempt to send all pending emails."""
        sent_emails = []

        for label, email_content in self._emails_to_send.items():
            email_subject = email_content['subject']
            message = email_content['contents']
            attachments = email_content['attachments']

            try:
                self._yag.send(
                    to=self._gmail_account,
                    subject=f'Intelligent Surveillance: {email_subject}',
                    contents=[message],
                    attachments=attachments

                )
                sent_emails.append(label)

            except gaierror:
                return

        for label in sent_emails:
            del self._emails_to_send[label]

        self._save_emails_to_send()

