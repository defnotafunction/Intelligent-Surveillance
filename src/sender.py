import yagmail
from dotenv import load_dotenv
import os
import requests

load_dotenv()

class GmailSender:
    def __init__(self) -> None:
        self.app_password = os.getenv('GMAIL_APP_PASSWORD')
        self.gmail_account = os.getenv('GMAIL_ACCOUNT')

        self.yag = yagmail.SMTP(self.gmail_account, self.app_password)

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
            self.yag.send(
                to=self.gmail_account,
                subject=f'Intelligent Surveillance: {email_subject}',
                contents=[message],
                attachments=file_paths

            )
        except requests.ConnectionError:
            return {
                'subject': f'Intelligent Surveillance: {email_subject}',
                'contents': message,
                'attachments': file_paths,
            }

