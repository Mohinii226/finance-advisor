"""
Outgoing email for verification and password-reset links. Three backends, chosen with MAIL_BACKEND:

  console (default)  logs the message. Development only: the log would contain live login links, so it is refused when
                     APP_ENV=production.
  smtp               any SMTP provider. Env: SMTP_HOST, SMTP_PORT (587), SMTP_USER, SMTP_PASSWORD, SMTP_FROM,
                     SMTP_SECURITY = starttls (default) | ssl | none
"""
from __future__ import annotations

import logging
import os
import smtplib
import ssl
from email.message import EmailMessage

log = logging.getLogger("finsight.mail")


class ConsoleMailer:
    def send(self, to: str, subject: str, body: str) -> None:
        log.warning("DEV MAIL to=%s subject=%s\n%s", to, subject, body)


class SmtpMailer:
    def __init__(self, host: str, port: int, user: str, password: str, sender: str, security: str = "starttls", timeout: int = 15):
        self.host, self.port, self.user, self.password = host, port, user, password
        self.sender, self.security, self.timeout = sender, security, timeout

    def send(self, to: str, subject: str, body: str) -> None:
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = self.sender, to, subject
        msg.set_content(body)
        ctx = ssl.create_default_context()
        if self.security == "ssl":
            with smtplib.SMTP_SSL(self.host, self.port, context=ctx, timeout=self.timeout) as s:
                if self.user:
                    s.login(self.user, self.password)
                s.send_message(msg)
            return
        with smtplib.SMTP(self.host, self.port, timeout=self.timeout) as s:
            s.ehlo()
            if self.security == "starttls":
                s.starttls(context=ctx)
                s.ehlo()
            if self.user:
                s.login(self.user, self.password)
            s.send_message(msg)


def build_mailer(env=os.environ):
    backend = env.get("MAIL_BACKEND", "console").strip().lower()
    if backend == "smtp":
        host, sender = env.get("SMTP_HOST", ""), env.get("SMTP_FROM", "")
        if not host or not sender:
            raise RuntimeError("MAIL_BACKEND=smtp needs SMTP_HOST and SMTP_FROM (and usually SMTP_USER / SMTP_PASSWORD).")
        return SmtpMailer(host, int(env.get("SMTP_PORT", "587")), env.get("SMTP_USER", ""), env.get("SMTP_PASSWORD", ""),
                          sender, env.get("SMTP_SECURITY", "starttls").strip().lower())
    if env.get("APP_ENV", "").strip().lower() == "production":
        raise RuntimeError("MAIL_BACKEND=console is not allowed when APP_ENV=production (it would log login links). Use smtp.")
    return ConsoleMailer()
