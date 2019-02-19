# send_process_report_email.py
# Import smtplib for the actual sending function
import smtplib
# Import the email modules we'll need
from email.mime.text import MIMEText
import sys


report_email_smtp_server = 'smtp.qq.com'
report_email_smtp_port = 465
report_email_sender = 'rainbowwu0625@foxmail.com'
report_email_password = 'nisrhhlwaoiicacf'  # !!!!!!!!!
report_email_receiver = ['rainbowwu0625@foxmail.com']


def send_email_main_process(subject, text):
    smtp_ssl_host = report_email_smtp_server
    # 'smtp.gmail.com'  # smtp.mail.yahoo.com
    smtp_ssl_port = report_email_smtp_port
    username = report_email_sender
    password = report_email_password
    sender = report_email_sender
    targets = report_email_receiver

    # Save content
    msg = MIMEText(text)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ', '.join(targets)

    server = smtplib.SMTP(smtp_ssl_host)
    # server.ehlo()
    # server.starttls()
    server.login(username, password)
    server.sendmail(sender, targets, msg.as_string())
    server.quit()

    print('Email of \'' + subject + '\' is sent!')

    return True


if __name__ == '__main__':
    # Set options
    subject = 'test'
    text = 'Hello world!'

    # Send email
    send_email_main_process(subject, text)
