# bot.py
from __future__ import annotations

from typing import Optional, Set, Tuple, Dict, List
from collections import deque
import heapq

from game_message import *


class Bot:
    def __init__(self):
        print("Initializing your super mega duper bot")

        # Defender spore (stays at base)
        self.defender_id: Optional[str] = None

        # Pending defender spawn info so we can identify the new spore
        self._pending_defender: Optional[dict] = None  # {"prev_ids": set[str], "biomass": int}

        # Track main spawner (the first one created)
        self.main_spawner_id: Optional[str] = None

    def get_next_move(self, game_message: TeamGameState) -> list[Action]:
        actions: list[Action] = []
        world: GameWorld = game_message.world
        my_team_id = game_message.yourTeamId
        my_team: TeamInfo = world.teamInfos[my_team_id]
        neutral_id = game_message.constants.neutralTeamId
        current_tick = game_message.tick

        # 1) Create initial spawner if none exist
        if len(my_team.spawners) == 0 and my_team.spores:
            if not self._has_spawner_at(world, my_team.spores[0].position):
                return [SporeCreateSpawnerAction(sporeId=my_team.spores[0].id)]
            return []

        # Track main spawner
        if self.main_spawner_id is None and my_team.spawners:
            self.main_spawner_id = my_team.spawners[0].id

        # Defender lifecycle
        self._update_defender_status(my_team)
        self._resolve_pending_defender(my_team)

        # 2) Spawn a defender once if missing
        if (
                self.defender_id is None
                and self._pending_defender is None
                and my_team.spawners
                and my_team.nutrients >= 4
        ):
            defender_biomass = max(1, my_team.nutrients // 5)
            prev_ids = {s.id for s in my_team.spores}

            actions.append(
                SpawnerProduceSporeAction(
                    spawnerId=my_team.spawners[0].id,
                    biomass=defender_biomass,
                )
            )
            self._pending_defender = {"prev_ids": prev_ids, "biomass": defender_biomass}
            return actions

        # 3) Non-main spawners produce spores on even turns
        is_even_turn = (current_tick % 2 == 0)
        if is_even_turn:
            for spawner in my_team.spawners:
                if spawner.id != self.main_spawner_id and my_team.nutrients >= 1:
                    actions.append(
                        SpawnerProduceSporeAction(
                            spawnerId=spawner.id,
                            biomass=1,
                        )
                    )

        # Non-defender spores
        available_spores = [s for s in my_team.spores if s.id != self.defender_id]

        # If we only have defender, spawn an attacker
        if not available_spores and my_team.spawners and my_team.nutrients >= 20:
            actions.append(
                SpawnerProduceSporeAction(
                    spawnerId=my_team.spawners[0].id,
                    biomass=max(20, my_team.nutrients // 3),
                )
            )
            return actions

        # Build cost map once for all spores
        neutral_spores = self._get_neutral_spore_biomass(world, neutral_id)
        cost_map = self._build_cost_map(world, neutral_spores)

        # Get targets: prioritize nutrients -> empty -> enemy
        nutrient_tiles = self._get_nutrient_tiles(world, my_team_id)
        empty_tiles = self._get_empty_tiles(world, my_team_id, neutral_id, neutral_spores)
        enemy_tiles = self._get_enemy_tiles(world, my_team_id, neutral_id)

        # Reservations to avoid collisions
        reserved_destinations: Set[Tuple[int, int]] = set()
        assigned_targets: Set[Tuple[int, int]] = set()

        # 4) Move each spore using A* pathfinding
        for spore in available_spores:
            if spore.biomass <= 0:
                continue

            # Find best target for this spore: nutrients first, then empty, then enemy
            target = self._find_best_target_astar(
                spore=spore,
                world=world,
                cost_map=cost_map,
                neutral_spores=neutral_spores,
                nutrient_tiles=nutrient_tiles,
                empty_tiles=empty_tiles,
                enemy_tiles=enemy_tiles,
                assigned_targets=assigned_targets,
            )

            if target is None:
                continue

            assigned_targets.add((target.x, target.y))

            # Get the next move using A* (just one step)
            next_pos = self._get_next_move_astar(
                spore=spore,
                goal=target,
                world=world,
                cost_map=cost_map,
                neutral_spores=neutral_spores,
                reserved_destinations=reserved_destinations,
            )

            if next_pos is None:
                continue

            # If already at target, consider creating a spawner
            if next_pos.x == spore.position.x and next_pos.y == spore.position.y:
                if self._should_create_spawner_here(world, my_team, spore, spore.position):
                    actions.append(SporeCreateSpawnerAction(sporeId=spore.id))
                continue

            # Reserve destination
            reserved_destinations.add((next_pos.x, next_pos.y))

            # Use SporeMoveAction for single-step movement
            dx = next_pos.x - spore.position.x
            dy = next_pos.y - spore.position.y
            actions.append(SporeMoveAction(sporeId=spore.id, direction=Position(x=dx, y=dy)))

        # 5) Spawn additional spores if we have resources
        if len(available_spores) < 20 and my_team.spawners and my_team.nutrients >= 20:
            actions.append(
                SpawnerProduceSporeAction(
                    spawnerId=my_team.spawners[0].id,
                    biomass=max(20, my_team.nutrients // 3),
                )
            )

        print(actions)
        return actions

    # -----------------------------
    # Defender logic
    # -----------------------------
    def _update_defender_status(self, my_team: TeamInfo):
        if self.defender_id and not any(s.id == self.defender_id for s in my_team.spores):
            self.defender_id = None

    def _resolve_pending_defender(self, my_team: TeamInfo):
        if self._pending_defender is None:
            return

        prev_ids: Set[str] = self._pending_defender["prev_ids"]
        expected_biomass: int = self._pending_defender["biomass"]
        spawner_positions = {(sp.position.x, sp.position.y) for sp in my_team.spawners}

        new_spores = [s for s in my_team.spores if s.id not in prev_ids]
        for spore in new_spores:
            if (
                    (spore.position.x, spore.position.y) in spawner_positions
                    and spore.biomass == expected_biomass
            ):
                self.defender_id = spore.id
                print(f"Defender assigned: {spore.id}")
                self._pending_defender = None
                return

        if not new_spores:
            self._pending_defender = None

    # -----------------------------
    # Cost map and tile helpers
    # -----------------------------
    def _get_neutral_spore_biomass(self, world: GameWorld, neutral_id: str) -> Dict[Tuple[int, int], int]:
        """Returns dict of {(x, y): biomass} for neutral spores"""
        return {
            (s.position.x, s.position.y): s.biomass
            for s in world.spores
            if s.teamId == neutral_id
        }

    def _build_cost_map(
            self,
            world: GameWorld,
            neutral_spores: Dict[Tuple[int, int], int],
    ) -> List[List[float]]:
        """
        Build a cost map for A* pathfinding:
        - Empty tiles: weight = 1
        - Nutrient tiles: weight = 1 - nutrient_value (lower = more attractive, min 0.1)
        - Neutral spores: weight = biomass (expensive to go through)
        """
        w, h = world.map.width, world.map.height
        cost_map = [[1.0 for _ in range(h)] for _ in range(w)]

        for x in range(w):
            for y in range(h):
                # Base cost for empty tiles
                cost = 1.0

                # Nutrient tiles have negative-ish cost (more attractive)
                # We use 1 - nutrient_value, with a minimum to keep it positive
                nutrient_value = world.map.nutrientGrid[x][y]
                if nutrient_value > 0:
                    cost = max(0.1, 1.0 - nutrient_value)

                # Neutral spores are expensive (weight = biomass)
                if (x, y) in neutral_spores:
                    cost = float(neutral_spores[(x, y)])

                cost_map[x][y] = cost

        return cost_map

    def _get_nutrient_tiles(self, world: GameWorld, my_team_id: str) -> List[Position]:
        """Get all nutrient tiles not owned by us"""
        tiles: List[Position] = []
        for x in range(world.map.width):
            for y in range(world.map.height):
                if world.map.nutrientGrid[x][y] > 0 and world.ownershipGrid[x][y] != my_team_id:
                    tiles.append(Position(x=x, y=y))
        return tiles

    def _get_empty_tiles(
            self,
            world: GameWorld,
            my_team_id: str,
            neutral_id: str,
            neutral_spores: Dict[Tuple[int, int], int],
    ) -> List[Position]:
        """Get empty tiles (not owned, not neutral spores)"""
        tiles: List[Position] = []
        for x in range(world.map.width):
            for y in range(world.map.height):
                owner = world.ownershipGrid[x][y]
                if owner == "" or owner == neutral_id:
                    if (x, y) not in neutral_spores:
                        tiles.append(Position(x=x, y=y))
        return tiles

    def _get_enemy_tiles(
            self,
            world: GameWorld,
            my_team_id: str,
            neutral_id: str,
    ) -> List[Position]:
        """Get enemy-owned tiles"""
        tiles: List[Position] = []
        for x in range(world.map.width):
            for y in range(world.map.height):
                owner = world.ownershipGrid[x][y]
                if owner != "" and owner != my_team_id and owner != neutral_id:
                    tiles.append(Position(x=x, y=y))
        return tiles

    # -----------------------------
    # A* pathfinding
    # -----------------------------
    def _find_best_target_astar(
            self,
            spore: Spore,
            world: GameWorld,
            cost_map: List[List[float]],
            neutral_spores: Dict[Tuple[int, int], int],
            nutrient_tiles: List[Position],
            empty_tiles: List[Position],
            enemy_tiles: List[Position],
            assigned_targets: Set[Tuple[int, int]],
    ) -> Optional[Position]:
        """
        Find the best target for a spore using A* path cost.
        Priority: nutrients (lowest path cost) -> empty -> enemy
        """
        w, h = world.map.width, world.map.height

        # First try nutrient tiles
        best_target = None
        best_cost = float("inf")

        for target in nutrient_tiles:
            if (target.x, target.y) in assigned_targets:
                continue

            path_cost = self._astar_path_cost(
                width=w,
                height=h,
                start=spore.position,
                goal=target,
                cost_map=cost_map,
            )

            if path_cost is not None and path_cost < best_cost:
                best_cost = path_cost
                best_target = target

        if best_target is not None:
            return best_target

        # Then try empty tiles
        for target in empty_tiles:
            if (target.x, target.y) in assigned_targets:
                continue

            # Just use Manhattan distance for empty tiles (faster)
            dist = abs(target.x - spore.position.x) + abs(target.y - spore.position.y)
            if dist > 0 and dist < best_cost:
                best_cost = dist
                best_target = target

        if best_target is not None:
            return best_target

        # Finally try enemy tiles
        for target in enemy_tiles:
            if (target.x, target.y) in assigned_targets:
                continue

            dist = abs(target.x - spore.position.x) + abs(target.y - spore.position.y)
            if dist > 0 and dist < best_cost:
                best_cost = dist
                best_target = target

        return best_target

    def _astar_path_cost(
            self,
            width: int,
            height: int,
            start: Position,
            goal: Position,
            cost_map: List[List[float]],
    ) -> Optional[float]:
        """
        A* search to find the lowest total cost path from start to goal.
        Returns the total path cost, or None if unreachable.
        """
        s = (start.x, start.y)
        g = (goal.x, goal.y)

        if s == g:
            return 0.0

        def heur(x: int, y: int) -> float:
            return abs(x - g[0]) + abs(y - g[1])

        # Priority queue: (f_score, g_score, x, y)
        open_pq: List[Tuple[float, float, int, int]] = []
        heapq.heappush(open_pq, (heur(s[0], s[1]), 0.0, s[0], s[1]))

        g_score: Dict[Tuple[int, int], float] = {s: 0.0}

        while open_pq:
            f_cur, g_cur, x, y = heapq.heappop(open_pq)
            cur = (x, y)

            # Skip stale entries
            if g_cur != g_score.get(cur, float("inf")):
                continue

            if cur == g:
                return g_cur

            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue

                step_cost = cost_map[nx][ny]
                tentative_g = g_cur + step_cost

                if tentative_g < g_score.get((nx, ny), float("inf")):
                    g_score[(nx, ny)] = tentative_g
                    heapq.heappush(open_pq, (tentative_g + heur(nx, ny), tentative_g, nx, ny))

        return None  # Unreachable


    def _get_next_move_astar(
            self,
            spore: Spore,
            goal: Position,
            world: GameWorld,
            cost_map: List[List[float]],
            neutral_spores: Dict[Tuple[int, int], int],
            reserved_destinations: Set[Tuple[int, int]],
    ) -> Optional[Position]:
        """
        Get the next single move towards goal using A*.
        Returns the Position to move to (adjacent to current position), or None.
        """
        w, h = world.map.width, world.map.height
        start = spore.position

        if (start.x, start.y) == (goal.x, goal.y):
            return start

        s = (start.x, start.y)
        g = (goal.x, goal.y)

        def heur(x: int, y: int) -> float:
            return abs(x - g[0]) + abs(y - g[1])

        # Priority queue: (f_score, g_score, x, y)
        open_pq: List[Tuple[float, float, int, int]] = []
        heapq.heappush(open_pq, (heur(s[0], s[1]), 0.0, s[0], s[1]))

        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {s: None}
        g_score: Dict[Tuple[int, int], float] = {s: 0.0}

        while open_pq:
            f_cur, g_cur, x, y = heapq.heappop(open_pq)
            cur = (x, y)

            if g_cur != g_score.get(cur, float("inf")):
                continue

            if cur == g:
                break

            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if nx < 0 or ny < 0 or nx >= w or ny >= h:
                    continue

                step_cost = cost_map[nx][ny]
                tentative_g = g_cur + step_cost

                if tentative_g < g_score.get((nx, ny), float("inf")):
                    g_score[(nx, ny)] = tentative_g
                    came_from[(nx, ny)] = cur
                    heapq.heappush(open_pq, (tentative_g + heur(nx, ny), tentative_g, nx, ny))

        if g not in came_from:
            # Goal not reachable, try to move towards it anyway
            return self._get_best_adjacent_move(start, goal, w, h, neutral_spores, reserved_destinations)

        # Reconstruct path and get first step
        path: List[Tuple[int, int]] = []
        cur = g
        while cur is not None:
            path.append(cur)
            cur = came_from[cur]
        path.reverse()

        if len(path) < 2:
            return start

        # Get the first step (path[1])
        next_step = path[1]

        # Check if it's safe to move there (not reserved, not a neutral spore we want to avoid)
        if next_step in reserved_destinations:
            return self._get_best_adjacent_move(start, goal, w, h, neutral_spores, reserved_destinations)

        # Avoid landing on neutral spores if possible
        if next_step in neutral_spores:
            alt_move = self._get_best_adjacent_move(start, goal, w, h, neutral_spores, reserved_destinations)
            if alt_move and (alt_move.x, alt_move.y) not in neutral_spores:
                return alt_move
            # If no alternative, we may have to go through the neutral spore
            return Position(x=next_step[0], y=next_step[1])

        return Position(x=next_step[0], y=next_step[1])

    def _get_best_adjacent_move(
            self,
            start: Position,
            goal: Position,
            width: int,
            height: int,
            neutral_spores: Dict[Tuple[int, int], int],
            reserved_destinations: Set[Tuple[int, int]],
    ) -> Optional[Position]:
        """Get the best adjacent move towards goal, avoiding neutral spores and reserved spots"""
        x, y = start.x, start.y
        best_pos = None
        best_dist = float("inf")

        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            if (nx, ny) in reserved_destinations:
                continue
            if (nx, ny) in neutral_spores:
                continue

            dist = abs(nx - goal.x) + abs(ny - goal.y)
            if dist < best_dist:
                best_dist = dist
                best_pos = Position(x=nx, y=ny)

        return best_pos

    # -----------------------------
    # Spawner helpers
    # -----------------------------
    def _has_spawner_at(self, world: GameWorld, pos: Position) -> bool:
        return any(s.position.x == pos.x and s.position.y == pos.y for s in world.spawners)

    def _distance_to_nearest_own_spawner(self, my_team: TeamInfo, pos: Position) -> int:
        if not my_team.spawners:
            return 10 ** 9
        return min(
            abs(sp.position.x - pos.x) + abs(sp.position.y - pos.y)
            for sp in my_team.spawners
        )

    def _should_create_spawner_here(
            self,
            world: GameWorld,
            my_team: TeamInfo,
            spore: Spore,
            pos: Position
    ) -> bool:
        """
        Create spawner on nutrient tiles with minimum distance of 5 between spawners.
        """
        if spore.biomass < 2:
            return False

        if self._has_spawner_at(world, pos):
            return False

        # Check minimum distance to existing spawners (min 5)
        for spawner in my_team.spawners:
            dist = abs(spawner.position.x - pos.x) + abs(spawner.position.y - pos.y)
            if dist < 5:
                return False

        # Prioritize nutrient tiles
        has_nutrients = world.map.nutrientGrid[pos.x][pos.y] > 0
        if has_nutrients:
            return True

        # Otherwise only create if far from existing spawners
        return self._distance_to_nearest_own_spawner(my_team, pos) >= 6