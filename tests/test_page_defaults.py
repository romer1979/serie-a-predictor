"""Run with: venv/bin/python -m unittest discover -s tests -v."""

import atexit
import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from flask import template_rendered


# Importing the app creates tables: always use a disposable database.
test_directory = tempfile.TemporaryDirectory()
atexit.register(test_directory.cleanup)
with patch.dict(os.environ, {
    "DATABASE_URL": f"sqlite:///{test_directory.name}/defaults.db",
    "ADMIN_FORCE_RESET": "0",
}):
    import app as application


class FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        value = cls(2026, 10, 15, 12, tzinfo=timezone.utc)
        return value.astimezone(tz) if tz else value.replace(tzinfo=None)


class PageDefaultsTest(unittest.TestCase):
    def setUp(self):
        self.context = application.app.app_context()
        self.context.push()
        self.addCleanup(self.context.pop)
        application.db.drop_all()
        application.db.create_all()
        application.app.config["TESTING"] = True
        self.clock = patch.object(application, "datetime", FixedDateTime)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.refresh = patch.object(application, "update_fixtures_adaptive").start()
        self.addCleanup(patch.stopall)
        user = application.User(username="player", password_hash="unused", is_admin=True)
        application.db.session.add(user)
        application.db.session.flush()
        self.user_id = user.id
        self.fixture("2026-27", "1", "2026-10-14", finished=True)
        self.fixture("2026-27", "9", "2026-10-01", finished=True)
        scored = self.fixture("2026-27", "10", "2026-10-15", finished=True)
        self.fixture("2026-27", "10", "2026-10-16")
        self.fixture("2026-27", "11", "2026-10-22")
        self.fixture("2026-27", "38", "2027-05-25")
        self.fixture("2027-28", "1", "2027-08-20")
        self.fixture("Italian Serie A 2024/25", "38", "2025-05-25", finished=True)
        self.fixture("2025-26", "38", "2026-05-25", finished=True)
        wc = self.fixture("2026", "Final", "2026-07-19", finished=True, competition="WC")
        for fixture in (scored, wc):
            application.db.session.add(application.Prediction(
                user_id=user.id, fixture_id=fixture.id, selection="1",
            ))
        application.db.session.commit()
        self.client = application.app.test_client()
        with self.client.session_transaction() as session:
            session["_user_id"] = str(user.id)
            session["_fresh"] = True

    def tearDown(self):
        application.db.session.remove()

    def fixture(self, season, matchday, date, finished=False, competition="SA", status=None):
        fixture = application.Fixture(
            match_id=str(application.Fixture.query.count() + 1),
            match_date=datetime.fromisoformat(date),
            home_team="Home", away_team="Away", season=season, matchday=matchday,
            competition_code=competition,
            status=status or ("FINISHED" if finished else "SCHEDULED"),
            home_score=2 if finished else None, away_score=0 if finished else None,
        )
        application.db.session.add(fixture)
        application.db.session.flush()
        return fixture

    def page(self, url):
        rendered = []

        def capture(sender, template, context, **extra):
            rendered.append(context)

        with template_rendered.connected_to(capture, application.app):
            response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        return response.get_data(as_text=True), rendered[-1]

    def test_default_pages_select_current_serie_a_and_refresh(self):
        for url in ("/leaderboard", "/history", "/admin/coverage", "/admin/results"):
            with self.subTest(url=url):
                self.refresh.reset_mock()
                _, context = self.page(url)
                self.assertEqual(context["season"], "2026-27")
                self.assertEqual(context.get("competition_code", context["current_competition"]["code"]), "SA")
                self.refresh.assert_called_once_with()
                if url == "/leaderboard":
                    self.assertEqual(context["scope"], "season")
                    self.assertEqual(context["users"], [{"username": "player", "points": 1}])
                else:
                    self.assertEqual(context["matchday"], "10")

    def test_weekly_leaderboard_includes_partial_round(self):
        _, context = self.page("/leaderboard?scope=week")
        self.assertEqual(context["matchday"], "10")
        self.assertEqual(context["users"], [{"username": "player", "points": 1}])

    def test_admin_moves_to_next_round_after_all_matches_finish(self):
        application.Fixture.query.filter_by(season="2026-27", matchday="10").update({
            "status": "FINISHED", "home_score": 2, "away_score": 0,
        })
        application.db.session.commit()
        for url in ("/admin/coverage", "/admin/results"):
            with self.subTest(url=url):
                _, context = self.page(url)
                self.assertEqual(context["matchday"], "11")
                # Completed rounds remain available when explicitly selected.
                _, context = self.page(url + "?matchday=10")
                self.assertEqual(context["matchday"], "10")
        _, context = self.page("/history")
        self.assertEqual(context["matchday"], "10")

    def test_admin_keeps_ongoing_round_even_when_live_scores_are_present(self):
        for status in ("IN_PLAY", "PAUSED"):
            application.Fixture.query.filter_by(season="2026-27", matchday="10").update({
                "status": status, "home_score": 1, "away_score": 1,
            })
            application.db.session.commit()
            for url in ("/admin/coverage", "/admin/results"):
                with self.subTest(status=status, url=url):
                    _, context = self.page(url)
                    self.assertEqual(context["matchday"], "10")

    def test_admin_ignores_postponed_matches_in_completed_round(self):
        application.Fixture.query.filter_by(season="2026-27", matchday="10").update({
            "status": "FINISHED", "home_score": 2, "away_score": 0,
        })
        self.fixture("2026-27", "10", "2026-10-16", status="POSTPONED")
        application.db.session.commit()
        for url in ("/admin/coverage", "/admin/results"):
            with self.subTest(url=url):
                _, context = self.page(url)
                self.assertEqual(context["matchday"], "11")

    def test_admin_does_not_treat_finished_match_with_missing_score_as_upcoming(self):
        application.Fixture.query.filter_by(season="2026-27", matchday="10").update({
            "status": "FINISHED", "home_score": None, "away_score": None,
        })
        application.db.session.commit()
        for url in ("/admin/coverage", "/admin/results"):
            with self.subTest(url=url):
                _, context = self.page(url)
                self.assertEqual(context["matchday"], "11")

    def test_manual_season_and_round_selections_are_preserved(self):
        for url in ("/leaderboard?scope=week&", "/history?", "/admin/coverage?", "/admin/results?"):
            with self.subTest(url=url):
                _, context = self.page(url + "season=2025-26&matchday=38")
                self.assertEqual(context["season"], "2025-26")
                self.assertEqual(context["matchday"], "38")

    def test_stale_filters_fall_back_to_latest_round(self):
        for url in ("/leaderboard?scope=week&", "/history?", "/admin/coverage?", "/admin/results?"):
            with self.subTest(url=url):
                _, context = self.page(url + "season=missing&matchday=Final")
                self.assertEqual(context["season"], "2026-27")
                self.assertEqual(context["matchday"], "10")

    def test_world_cup_remains_available(self):
        for url in ("/world-cup/history", "/world-cup/leaderboard?scope=week",
                    "/admin/coverage?competition=WC", "/admin/results?competition=WC"):
            with self.subTest(url=url):
                _, context = self.page(url)
                self.assertEqual(context["season"], "2026")
                self.assertEqual(context["matchday"], "Final")

    def test_navigation_returns_to_serie_a_from_world_cup(self):
        html, _ = self.page("/world-cup/history")
        nav = html.split('id="primary-nav"')[1].split("</ul>")[0]
        self.assertIn('href="/leaderboard"', nav)
        self.assertIn('href="/history"', nav)
        self.assertNotIn('href="/world-cup/leaderboard"', nav)
        self.assertNotIn('href="/world-cup/history"', nav)

    def test_preseason_opens_first_round_not_final_round(self):
        _, context = self.page("/leaderboard?scope=week&season=2027-28")
        self.assertEqual(context["matchday"], "1")

    def test_postponed_round_does_not_replace_active_round(self):
        self.fixture("2026-27", "12", "2026-10-14", status="POSTPONED")
        application.db.session.commit()
        self.assertEqual(application.latest_active_matchday("2026-27"), "10")

    def test_current_season_wins_over_future_and_descriptive_labels(self):
        self.assertEqual(application.current_season_from_db(), "2026-27")
        application.Fixture.query.filter_by(season="2026-27").delete()
        application.db.session.commit()
        self.assertEqual(application.current_season_from_db(), "2025-26")

    def test_empty_database_renders_all_pages(self):
        application.Prediction.query.delete()
        application.Fixture.query.delete()
        application.db.session.commit()
        for url in ("/leaderboard", "/history", "/admin/coverage", "/admin/results"):
            with self.subTest(url=url):
                _, context = self.page(url)
                self.assertIsNone(context["season"])
                self.assertIsNone(context["matchday"])

    def test_non_admin_cannot_open_admin_pages(self):
        application.db.session.get(application.User, self.user_id).is_admin = False
        application.db.session.commit()
        for url in ("/admin/coverage", "/admin/results"):
            self.assertEqual(self.client.get(url).status_code, 403)


if __name__ == "__main__":
    unittest.main()
